from pathlib import Path
from typing import Dict, List, Optional
import math
import re

from tqdm import tqdm

from .config import AppConfig
from .logging_utils import get_logger
from .utils import append_jsonl, ensure_dir, format_command, read_json, read_jsonl, write_jsonl


logger = get_logger(__name__)
_JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_SENTENCE_END_RE = re.compile(r"[。！？.!?]+[\"'”’」』]?\\s*$")
_END_PUNCT = set("。！？.!?")
_END_QUOTES = set("\"'”’」』")
_COMMA_PUNCT = set("、，,;；")
_SPACE_RE = re.compile(r"[ \u3000]+")


def _is_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _join_text(a: str, b: str) -> str:
    if not a:
        return b
    if _JP_CHAR_RE.search(a) or _JP_CHAR_RE.search(b):
        return a + b
    if a.endswith(("-", "—")):
        return a + b
    return f"{a.rstrip()} {b.lstrip()}"


def _split_sentences(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        buf.append(ch)
        if ch in _END_PUNCT:
            j = i + 1
            while j < len(text) and text[j] in _END_QUOTES:
                buf.append(text[j])
                j += 1
            sentence = "".join(buf).strip()
            if sentence:
                parts.append(sentence)
            buf = []
            i = j
            continue
        i += 1
    rest = "".join(buf).strip()
    if rest:
        parts.append(rest)
    return parts


def _split_by_commas(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _COMMA_PUNCT:
            chunk = "".join(buf).strip()
            if chunk:
                parts.append(chunk)
            buf = []
    rest = "".join(buf).strip()
    if rest:
        parts.append(rest)
    return parts if parts else [text]


def _split_by_spaces(text: str, min_chars: int = 3) -> List[str]:
    raw = [p.strip() for p in _SPACE_RE.split(text) if p.strip()]
    if len(raw) <= 1:
        return [text]
    merged: List[str] = []
    for part in raw:
        if merged and len(part) < min_chars:
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)
    return merged


def _split_by_length(text: str, parts: int) -> List[str]:
    if parts <= 1:
        return [text]
    text = text.strip()
    if not text:
        return []
    size = max(1, math.ceil(len(text) / parts))
    out = []
    for i in range(0, len(text), size):
        chunk = text[i : i + size].strip()
        if chunk:
            out.append(chunk)
    return out


def _split_segment_by_sentence(
    segment: Dict[str, object], config: AppConfig
) -> List[Dict[str, object]]:
    text = str(segment.get("text", "")).strip()
    if not text:
        return []
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    duration = max(0.001, end - start)

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]
    if len(sentences) == 1 and _SPACE_RE.search(text):
        sentences = _split_by_spaces(text)

    refined: List[str] = []
    total_len = max(1, sum(len(s) for s in sentences))
    for sent in sentences:
        predicted = duration * (len(sent) / total_len)
        if predicted > config.audio.max_segment_sec:
            chunks = _split_by_commas(sent)
            if len(chunks) == 1 and predicted > config.audio.max_segment_sec:
                parts = max(2, math.ceil(predicted / config.audio.max_segment_sec))
                chunks = _split_by_length(sent, parts)
            refined.extend(chunks)
        else:
            refined.append(sent)

    if not refined:
        refined = [text]

    total_len = max(1, sum(len(s) for s in refined))
    results: List[Dict[str, object]] = []
    cur = start
    for idx, sent in enumerate(refined):
        weight = len(sent) / total_len
        seg_dur = duration * weight
        seg_end = cur + seg_dur
        if idx == len(refined) - 1:
            seg_end = end
        results.append(
            {
                "start": cur,
                "end": seg_end,
                "text": sent,
                "confidence": float(segment.get("confidence", 0.0)),
            }
        )
        cur = seg_end
    return results


def _merge_asr_segments(segments: List[Dict[str, object]], config: AppConfig) -> List[Dict[str, object]]:
    if not segments:
        return []
    merged: List[Dict[str, object]] = []
    cur: Optional[Dict[str, object]] = None
    conf_sum = 0.0
    dur_sum = 0.0

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = max(0.0, end - start)
        conf = float(seg.get("confidence", 0.0))

        if cur is None:
            cur = {"start": start, "end": end, "text": text}
            conf_sum = conf * dur
            dur_sum = dur
            continue

        gap = start - float(cur["end"])
        cur_len = float(cur["end"]) - float(cur["start"])
        should_break = gap > config.audio.merge_pause_sec
        if cur_len >= config.audio.min_segment_sec and cur_len + dur > config.audio.max_segment_sec:
            should_break = True

        if should_break:
            merged.append(
                {
                    "start": float(cur["start"]),
                    "end": float(cur["end"]),
                    "text": str(cur["text"]),
                    "confidence": conf_sum / max(0.001, dur_sum),
                }
            )
            cur = {"start": start, "end": end, "text": text}
            conf_sum = conf * dur
            dur_sum = dur
            continue

        cur["text"] = _join_text(str(cur["text"]), text)
        cur["end"] = end
        conf_sum += conf * dur
        dur_sum += dur
        cur_len = float(cur["end"]) - float(cur["start"])
        if _is_sentence_end(text) and cur_len >= config.audio.min_segment_sec:
            merged.append(
                {
                    "start": float(cur["start"]),
                    "end": float(cur["end"]),
                    "text": str(cur["text"]),
                    "confidence": conf_sum / max(0.001, dur_sum),
                }
            )
            cur = None
            conf_sum = 0.0
            dur_sum = 0.0

    if cur is not None:
        merged.append(
            {
                "start": float(cur["start"]),
                "end": float(cur["end"]),
                "text": str(cur["text"]),
                "confidence": conf_sum / max(0.001, dur_sum),
            }
        )

    return merged


def _split_segments(segments: List[Dict[str, object]], config: AppConfig) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for seg in segments:
        results.extend(_split_segment_by_sentence(seg, config))
    return results


def run_asr(
    chunks_dir: Path,
    manifest_path: Path,
    workdir: Path,
    config: AppConfig,
    language: Optional[str] = None,
) -> Path:
    ensure_dir(workdir)
    asr_path = workdir / "asr_segments.jsonl"
    raw_path = workdir / "asr_raw.jsonl"
    words_path = workdir / "asr_words.jsonl"

    need_segments = not asr_path.exists()
    need_words = config.asr.word_timestamps and not words_path.exists()
    if not need_segments and not need_words:
        logger.info("Using existing ASR segments: %s", asr_path)
        return asr_path

    manifest = read_json(manifest_path)
    segments_all: List[Dict[str, object]] = []
    words_all: List[Dict[str, object]] = []

    for entry in tqdm(manifest, desc="ASR", unit="chunk"):
        chunk_file = chunks_dir / entry["chunk"]
        chunk_out = chunks_dir / f"{chunk_file.stem}_asr.jsonl"
        segs = []
        has_words = False
        if chunk_out.exists():
            segs = read_jsonl(chunk_out)
            has_words = any("words" in seg for seg in segs)

        if (not chunk_out.exists()) or (need_words and not has_words):
            cmd = format_command(
                config.asr.command_template,
                audio=str(chunk_file),
                out=str(chunk_out),
                language=language or "",
                device=config.asr.device,
            )
            logger.info("Running ASR: %s", chunk_file.name)
            from .utils import run_command

            run_command(cmd)
            segs = read_jsonl(chunk_out)
            has_words = any("words" in seg for seg in segs)
        if need_segments:
            for seg in segs:
                seg_start = float(seg["start"]) + float(entry["start_sec"])
                seg_end = float(seg["end"]) + float(entry["start_sec"])
                segments_all.append(
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg.get("text", "").strip(),
                        "confidence": float(seg.get("confidence", 0.0)),
                    }
                )

        if need_words and has_words:
            for seg in segs:
                for word in seg.get("words", []) or []:
                    w_text = str(word.get("word", "")).strip()
                    if not w_text:
                        continue
                    words_all.append(
                        {
                            "start": float(word.get("start", 0.0)) + float(entry["start_sec"]),
                            "end": float(word.get("end", 0.0)) + float(entry["start_sec"]),
                            "word": w_text,
                        }
                    )

    if need_segments:
        segments_all.sort(key=lambda x: x["start"])
        if not raw_path.exists():
            write_jsonl(raw_path, segments_all)
        merged = _merge_asr_segments(segments_all, config)
        split_segments = _split_segments(merged, config)
        logger.info(
            "ASR segments merged: %s -> %s -> %s",
            len(segments_all),
            len(merged),
            len(split_segments),
        )
        append_jsonl(asr_path, split_segments)
    elif not raw_path.exists() and segments_all:
        write_jsonl(raw_path, segments_all)

    if need_words and words_all:
        words_all.sort(key=lambda x: x["start"])
        write_jsonl(words_path, words_all)
    return asr_path
