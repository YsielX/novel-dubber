import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import AppConfig
from .logging_utils import get_logger
from .utils import read_jsonl, write_jsonl


logger = get_logger(__name__)
_KEEP_RE = re.compile(r"[0-9a-zA-Z\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+")

try:
    from pykakasi import kakasi  # type: ignore

    _KAKASI = kakasi()
    _KAKASI.setMode("J", "H")
    _KAKASI.setMode("K", "H")
    _KAKASI.setMode("H", "H")
    _KAKASI.setMode("r", "Hepburn")
    _KAKASI.setMode("s", True)
    _KAKASI.setMode("C", True)
except Exception:  # pragma: no cover - optional dependency
    _KAKASI = None


def _to_kana(text: str) -> str:
    if not _KAKASI:
        return text
    converted = _KAKASI.convert(text)
    return "".join(item.get("hira", "") for item in converted)


def _normalize(text: str, use_kana: bool) -> str:
    text = text.lower()
    if use_kana:
        text = _to_kana(text)
    parts = _KEEP_RE.findall(text)
    return "".join(parts)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _best_match_words(
    text: str,
    words: List[Dict[str, object]],
    words_norm: List[str],
    start_idx: int,
    search_window: int,
    max_words: int,
    use_kana: bool,
) -> Optional[Tuple[int, int, float, str]]:
    if start_idx >= len(words_norm):
        return None
    end_limit = min(len(words_norm), start_idx + search_window)
    norm_text = _normalize(text, use_kana)
    if not norm_text:
        return None

    best: Optional[Tuple[int, int, float, str]] = None
    for i in range(start_idx, end_limit):
        combined: List[str] = []
        for j in range(i, min(end_limit, i + max_words)):
            combined.append(words_norm[j])
            cand_text = "".join(combined)
            score = _similarity(norm_text, cand_text)
            if best is None or score > best[2]:
                best = (i, j, score, "".join(str(w.get("word", "")) for w in words[i : j + 1]))
            if score >= 0.98:
                return best
    return best


def _best_match_segments(
    text: str,
    segments: List[Dict[str, object]],
    start_idx: int,
    search_window: int,
    max_combine: int,
    use_kana: bool,
) -> Optional[Tuple[int, int, float, str]]:
    if start_idx >= len(segments):
        return None
    end_limit = min(len(segments), start_idx + search_window)
    norm_text = _normalize(text, use_kana)
    if not norm_text:
        return None

    best: Optional[Tuple[int, int, float, str]] = None
    for i in range(start_idx, end_limit):
        combined: List[str] = []
        for j in range(i, min(end_limit, i + max_combine)):
            combined.append(str(segments[j].get("text", "")))
            cand_text = "".join(combined)
            score = _similarity(norm_text, _normalize(cand_text, use_kana))
            if best is None or score > best[2]:
                best = (i, j, score, cand_text)
            if score >= 0.98:
                return best
    return best


def _find_text_offset(
    text_segments: List[Dict[str, object]],
    words_norm: List[str],
    config: AppConfig,
    use_kana: bool,
) -> Tuple[int, float]:
    if not text_segments or not words_norm:
        return 0, 0.0
    window_size = max(1, config.alignment.offset_window_segments)
    audio_words = words_norm[: max(1, config.alignment.offset_search_words)]
    audio_norm = "".join(audio_words)
    if not audio_norm:
        return 0, 0.0

    best_idx = 0
    best_score = 0.0
    limit = max(0, len(text_segments) - window_size)
    for i in range(0, limit + 1):
        window_text = "".join(str(seg.get("text", "")) for seg in text_segments[i : i + window_size])
        window_norm = _normalize(window_text, use_kana)
        if not window_norm:
            continue
        score = _similarity(window_norm, audio_norm)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def align_text_to_asr(
    text_segments_path: Path,
    asr_segments_path: Path,
    out_path: Path,
    config: AppConfig,
) -> Path:
    if out_path.exists():
        logger.info("Using existing alignment: %s", out_path)
        return out_path

    text_segments = read_jsonl(text_segments_path)
    words_path = asr_segments_path.parent / "asr_words.jsonl"
    if words_path.exists():
        return align_text_to_words(text_segments, words_path, out_path, config)

    asr_segments = read_jsonl(asr_segments_path)
    if not text_segments or not asr_segments:
        write_jsonl(out_path, text_segments)
        return out_path

    asr_segments.sort(key=lambda x: float(x.get("start", 0.0)))
    search_window = max(1, config.alignment.search_window)
    max_combine = max(1, config.alignment.max_combine)
    min_score = float(config.alignment.min_score)
    advance_on_miss = max(0, config.alignment.advance_on_miss)
    use_kana = bool(config.alignment.use_kana)
    if use_kana and _KAKASI is None:
        logger.warning("pykakasi not installed; kana normalization disabled")
        use_kana = False

    results: List[Dict[str, object]] = []
    cursor = 0
    for seg in text_segments:
        match = _best_match_segments(
            str(seg.get("text", "")),
            asr_segments,
            cursor,
            search_window,
            max_combine,
            use_kana,
        )
        if match and match[2] >= min_score:
            start_idx, end_idx, score, match_text = match
            start_ts = float(asr_segments[start_idx].get("start", 0.0))
            end_ts = float(asr_segments[end_idx].get("end", start_ts))
            out = dict(seg)
            out.update(
                {
                    "start": start_ts,
                    "end": end_ts,
                    "aligned": True,
                    "match_score": round(score, 4),
                    "asr_start_idx": start_idx,
                    "asr_end_idx": end_idx,
                    "asr_text": match_text,
                }
            )
            results.append(out)
            cursor = end_idx + 1
            continue

        out = dict(seg)
        out.update(
            {
                "aligned": False,
                "match_score": round(match[2], 4) if match else 0.0,
                "asr_start_idx": -1,
                "asr_end_idx": -1,
                "asr_text": match[3] if match else "",
            }
        )
        results.append(out)
        if advance_on_miss:
            cursor = min(len(asr_segments), cursor + advance_on_miss)

    write_jsonl(out_path, results)
    return out_path


def align_text_to_words(
    text_segments: List[Dict[str, object]],
    words_path: Path,
    out_path: Path,
    config: AppConfig,
) -> Path:
    words = read_jsonl(words_path)
    if not text_segments or not words:
        write_jsonl(out_path, text_segments)
        return out_path

    words.sort(key=lambda x: float(x.get("start", 0.0)))
    use_kana = bool(config.alignment.use_kana)
    if use_kana and _KAKASI is None:
        logger.warning("pykakasi not installed; kana normalization disabled")
        use_kana = False

    filtered_words: List[Dict[str, object]] = []
    words_norm: List[str] = []
    for w in words:
        norm = _normalize(str(w.get("word", "")), use_kana)
        if not norm:
            continue
        filtered_words.append(w)
        words_norm.append(norm)
    words = filtered_words
    if not words_norm:
        write_jsonl(out_path, text_segments)
        return out_path

    offset_idx, offset_score = _find_text_offset(text_segments, words_norm, config, use_kana)
    logger.info("Alignment offset: text_index=%s score=%.3f", offset_idx, offset_score)

    search_window = max(1, config.alignment.search_window)
    max_combine = max(1, config.alignment.max_combine)
    min_score = float(config.alignment.min_score)
    advance_on_miss = max(0, config.alignment.advance_on_miss)

    results: List[Dict[str, object]] = []
    cursor = 0
    for idx, seg in enumerate(text_segments):
        if idx < offset_idx:
            out = dict(seg)
            out.update(
                {
                    "aligned": False,
                    "match_score": 0.0,
                    "word_start_idx": -1,
                    "word_end_idx": -1,
                    "asr_text": "",
                }
            )
            results.append(out)
            continue

        match = _best_match_words(
            str(seg.get("text", "")),
            words,
            words_norm,
            cursor,
            search_window,
            max_combine,
            use_kana,
        )
        if match and match[2] >= min_score:
            start_idx, end_idx, score, match_text = match
            start_ts = float(words[start_idx].get("start", 0.0))
            end_ts = float(words[end_idx].get("end", start_ts))
            out = dict(seg)
            out.update(
                {
                    "start": start_ts,
                    "end": end_ts,
                    "aligned": True,
                    "match_score": round(score, 4),
                    "word_start_idx": start_idx,
                    "word_end_idx": end_idx,
                    "asr_text": match_text,
                }
            )
            results.append(out)
            cursor = end_idx + 1
            continue

        out = dict(seg)
        out.update(
            {
                "aligned": False,
                "match_score": round(match[2], 4) if match else 0.0,
                "word_start_idx": -1,
                "word_end_idx": -1,
                "asr_text": match[3] if match else "",
            }
        )
        results.append(out)
        if advance_on_miss:
            cursor = min(len(words_norm), cursor + advance_on_miss)

    write_jsonl(out_path, results)
    return out_path


def merge_alignment(
    labeled_path: Path,
    aligned_path: Path,
    out_path: Path,
) -> Path:
    if out_path.exists():
        logger.info("Using existing merged alignment: %s", out_path)
        return out_path

    labeled = read_jsonl(labeled_path)
    aligned = {str(seg.get("segment_id")): seg for seg in read_jsonl(aligned_path)}
    merged: List[Dict[str, object]] = []
    for seg in labeled:
        seg_id = str(seg.get("segment_id"))
        out = dict(seg)
        if seg_id in aligned:
            align = aligned[seg_id]
            for key in (
                "start",
                "end",
                "aligned",
                "match_score",
                "asr_start_idx",
                "asr_end_idx",
                "word_start_idx",
                "word_end_idx",
                "asr_text",
            ):
                if key in align:
                    out[key] = align[key]
        merged.append(out)

    write_jsonl(out_path, merged)
    return out_path
