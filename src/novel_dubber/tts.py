import base64
import hashlib
import json
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from .config import AppConfig
from .logging_utils import get_logger
from .utils import (
    ensure_dir,
    format_command,
    normalize_path,
    read_json,
    read_jsonl,
    run_command,
    write_jsonl,
)


logger = get_logger(__name__)
_REF_MIN_SEC = 3.0
_REF_MAX_SEC = 30.0
_REF_TARGET_SEC = 6.0
_TTS_RETRIES = 3


def _normalize_voice_map(voice_map: Dict[str, object]) -> Dict[str, List[Dict[str, str]]]:
    normalized: Dict[str, List[Dict[str, str]]] = {}
    for character, data in voice_map.items():
        if isinstance(data, dict) and "refs" in data:
            refs = data.get("refs", [])
            normalized[character] = [
                {"audio": str(ref["audio"]), "text": str(ref["text"])} for ref in refs
            ]
        elif isinstance(data, dict) and "ref_audio" in data:
            normalized[character] = [
                {"audio": str(data["ref_audio"]), "text": str(data.get("ref_text", ""))}
            ]
    return normalized


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return 0.0


def _trim_ref_audio(
    ref_audio: str, cache_dir: Path, start_sec: float, duration_sec: float
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{ref_audio}:{start_sec:.3f}:{duration_sec:.3f}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    out = cache_dir / f"ref_trim_{digest}.wav"
    if out.exists():
        return str(out)
    if out.exists():
        return str(out)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, start_sec)),
        "-t",
        str(max(0.1, duration_sec)),
        "-i",
        normalize_path(Path(ref_audio)),
        "-ac",
        "1",
        normalize_path(out),
    ]
    run_command(cmd)
    return str(out)



def _select_ref(refs: List[Dict[str, str]], cache_dir: Path) -> Tuple[str, str]:
    if not refs:
        return "", ""
    scored: List[Tuple[float, Dict[str, str], float]] = []
    for ref in refs:
        audio_path = Path(ref["audio"])
        dur = _wav_duration(audio_path)
        if dur <= 0:
            continue
        score = abs(dur - _REF_TARGET_SEC)
        scored.append((score, ref, dur))

    in_range = [item for item in scored if _REF_MIN_SEC <= item[2] <= _REF_MAX_SEC]
    if in_range:
        in_range.sort(key=lambda x: x[0])
        ref = in_range[0][1]
        return ref["audio"], ref.get("text", "")

    if scored:
        scored.sort(key=lambda x: x[2], reverse=True)
        ref = scored[0][1]
        dur = scored[0][2]
        if dur > _REF_MAX_SEC:
            start = max(0.0, (dur - _REF_MAX_SEC) / 2.0)
            trimmed = _trim_ref_audio(ref["audio"], cache_dir, start, _REF_MAX_SEC)
            logger.info("Trimmed ref audio to %.1fs: %s", _REF_MAX_SEC, trimmed)
            return trimmed, ref.get("text", "")
        return ref["audio"], ref.get("text", "")

    ref = refs[0]
    return ref["audio"], ref.get("text", "")


def _http_tts(
    config: AppConfig,
    text: str,
    ref_audio: str,
    ref_text: str,
    target_lang: str,
    out_path: Path,
) -> None:
    if not config.tts.http:
        raise RuntimeError("TTS HTTP config missing")

    template = config.tts.http.json_body_template
    payload = {}
    for key, val in template.items():
        payload[key] = val.format(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            target_lang=target_lang,
        )

    resp = None
    for attempt in range(_TTS_RETRIES):
        resp = requests.post(config.tts.http.endpoint, json=payload, timeout=600)
        if resp.status_code < 500:
            break
        if attempt < _TTS_RETRIES - 1:
            time.sleep(1.0 * (2**attempt))

    if resp is None:
        raise RuntimeError("TTS HTTP failed without response")
    if resp.status_code >= 400:
        raise RuntimeError(f"TTS HTTP {resp.status_code}: {resp.text[:400]}")
    content_type = resp.headers.get("Content-Type", "")

    if "application/json" in content_type:
        data = resp.json()
        if "audio_path" in data:
            out_path.write_bytes(Path(data["audio_path"]).read_bytes())
            return
        if "audio_base64" in data:
            out_path.write_bytes(base64.b64decode(data["audio_base64"]))
            return
        raise RuntimeError(f"Unknown JSON response from TTS: {json.dumps(data)[:200]}")

    out_path.write_bytes(resp.content)


def _cli_tts(
    config: AppConfig,
    text: str,
    ref_audio: str,
    ref_text: str,
    target_lang: str,
    out_path: Path,
) -> None:
    if not config.tts.cli:
        raise RuntimeError("TTS CLI config missing")

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
        tf.write(text)
        text_file = tf.name

    try:
        cmd = format_command(
            config.tts.cli.command_template,
            text=text,
            text_file=text_file,
            ref_audio=ref_audio,
            ref_text=ref_text,
            target_lang=target_lang,
            out=str(out_path),
        )
        run_command(cmd)
    finally:
        Path(text_file).unlink(missing_ok=True)


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _prepare_ref_audio(ref_audio: str, cache_dir: Path) -> str:
    if _is_ascii(ref_audio):
        return ref_audio
    src = Path(ref_audio)
    if not src.exists():
        return ref_audio
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(ref_audio.encode("utf-8")).hexdigest()[:12]
    dst = cache_dir / f"ref_{digest}{src.suffix or '.wav'}"
    if not dst.exists():
        shutil.copyfile(src, dst)
    return str(dst)


def _log_tts_error(out_dir: Path, seg: Dict[str, object], ref_audio: str, ref_text: str, err: Exception) -> None:
    from .utils import append_jsonl

    log_path = out_dir.parent / "tts_errors.jsonl"
    append_jsonl(
        log_path,
        [
            {
                "segment_id": seg.get("segment_id") or seg.get("group_id"),
                "character": seg.get("character"),
                "text": str(seg.get("translation", seg.get("text", "")))[:500],
                "ref_audio": ref_audio,
                "ref_text": str(ref_text)[:500],
                "error": str(err),
            }
        ],
    )


def _has_spoken_content(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _sanitize_tts_text(text: str) -> str:
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2026": "...",
        "\u2014": "-",
        "\u2013": "-",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return " ".join(out.split())


def _build_tts_groups(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    def _flush() -> None:
        nonlocal current
        if current is None:
            return
        groups.append(current)
        current = None

    for seg in segments:
        seg_id = str(seg.get("segment_id"))
        character = str(seg.get("character", "NARRATOR"))
        raw_text = str(seg.get("translation", seg.get("text", ""))).strip()
        start = seg.get("start")
        end = seg.get("end")
        text_index = seg.get("text_index")

        if not raw_text or not _has_spoken_content(raw_text):
            _flush()
            solo: Dict[str, object] = {
                "group_id": f"group_{seg_id}",
                "segment_ids": [seg_id],
                "character": character,
                "text": raw_text,
            }
            if start is not None:
                solo["start"] = start
            if end is not None:
                solo["end"] = end
            if text_index is not None:
                solo["text_index"] = text_index
            groups.append(solo)
            continue

        if current and current.get("character") == character:
            current["segment_ids"].append(seg_id)
            current["texts"].append(raw_text)
            if start is not None and current.get("start") is None:
                current["start"] = start
            if end is not None:
                current["end"] = end
            continue

        _flush()
        current = {
            "group_id": f"group_{seg_id}",
            "segment_ids": [seg_id],
            "character": character,
            "texts": [raw_text],
        }
        if start is not None:
            current["start"] = start
        if end is not None:
            current["end"] = end
        if text_index is not None:
            current["text_index"] = text_index

    _flush()

    for group in groups:
        texts = group.pop("texts", None)
        if isinstance(texts, list):
            group["text"] = "\n".join(texts)
    return groups


def _infer_silence_duration(seg: Dict[str, object], config: AppConfig) -> float:
    if "pause_after_sec" in seg:
        return max(0.1, float(seg.get("pause_after_sec", 0.0)))
    if "start" in seg and "end" in seg:
        try:
            return max(0.1, min(1.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))))
        except Exception:
            pass
    return max(0.1, float(config.text_dub.default_pause_sec))


def _write_silence(out_path: Path, duration_sec: float, sample_rate: int) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration_sec * sample_rate)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)


def _is_invalid_text_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "valid text" in msg or "invalid text" in msg or "text is empty" in msg


def _is_server_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "tts http 5" in msg or "http 5" in msg


def synthesize_segments(
    segments_path: Path,
    voice_map_path: Path,
    out_dir: Path,
    config: AppConfig,
    target_lang: str,
) -> Path:
    ensure_dir(out_dir)
    segments = read_jsonl(segments_path)
    voice_map = _normalize_voice_map(read_json(voice_map_path))
    groups = _build_tts_groups(segments)
    groups_path = out_dir / "tts_groups.jsonl"
    write_jsonl(groups_path, groups)

    for group in tqdm(groups, desc="TTS", unit="segment"):
        seg_id = str(group.get("group_id"))
        out_path = out_dir / f"{seg_id}.wav"
        if out_path.exists():
            continue

        character = str(group.get("character", "NARRATOR"))
        refs = voice_map.get(character) or voice_map.get("NARRATOR")
        if not refs:
            logger.warning("Missing voice ref for character: %s", character)
            continue
        ref_audio, ref_text = _select_ref(refs, out_dir.parent / "tts_ref_cache")
        if not ref_audio:
            logger.warning("Empty ref for character: %s", character)
            continue

        ref_audio = _prepare_ref_audio(ref_audio, out_dir.parent / "tts_ref_cache")
        ref_audio = normalize_path(Path(ref_audio))
        if not Path(ref_audio).exists():
            err = RuntimeError(f"Ref audio not found: {ref_audio}")
            _log_tts_error(out_dir, group, ref_audio, ref_text, err)
            _write_silence(
                out_path,
                _infer_silence_duration(group, config),
                config.audio.sample_rate,
            )
            continue

        text = str(group.get("text", "")).strip()
        text = _sanitize_tts_text(text)
        if not text or not _has_spoken_content(text):
            _write_silence(
                out_path,
                _infer_silence_duration(group, config),
                config.audio.sample_rate,
            )
            continue

        try:
            if config.tts.mode == "http":
                _http_tts(config, text, ref_audio, ref_text, target_lang, out_path)
            else:
                _cli_tts(config, text, ref_audio, ref_text, target_lang, out_path)
        except Exception as exc:
            if _is_invalid_text_error(exc):
                _log_tts_error(out_dir, group, ref_audio, ref_text, exc)
                _write_silence(
                    out_path,
                    _infer_silence_duration(group, config),
                    config.audio.sample_rate,
                )
                continue
            if _is_server_error(exc):
                _log_tts_error(out_dir, group, ref_audio, ref_text, exc)
                _write_silence(
                    out_path,
                    _infer_silence_duration(group, config),
                    config.audio.sample_rate,
                )
                continue
            _log_tts_error(out_dir, group, ref_audio, ref_text, exc)
            raise

    return out_dir
