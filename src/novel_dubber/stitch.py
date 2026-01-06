import wave
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import AppConfig
from .logging_utils import get_logger
from .utils import ensure_dir, normalize_path, read_jsonl, run_command


logger = get_logger(__name__)


def _normalize_wav(in_path: Path, out_path: Path, sample_rate: int) -> None:
    if out_path.exists():
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        normalize_path(in_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        normalize_path(out_path),
    ]
    run_command(cmd)


def _get_pause_after(segments: List[Dict[str, object]], idx: int, config: AppConfig) -> float:
    seg = segments[idx]
    if "pause_after_sec" in seg:
        return float(seg["pause_after_sec"])
    if idx == len(segments) - 1:
        return 0.0
    if "start" not in seg or "end" not in seg:
        return config.text_dub.default_pause_sec
    next_seg = segments[idx + 1]
    pause = float(next_seg.get("start", 0.0)) - float(seg.get("end", 0.0))
    return max(0.0, min(config.audio.max_pause_sec, pause))


def stitch_segments(
    segments_path: Path,
    tts_dir: Path,
    out_wav: Path,
    out_mp3: Path,
    config: AppConfig,
) -> Path:
    segments = read_jsonl(segments_path)
    groups_path = tts_dir / "tts_groups.jsonl"
    if groups_path.exists():
        groups = read_jsonl(groups_path)
        if groups:
            segments = groups
    if not segments:
        raise RuntimeError("No segments to stitch")

    has_timestamps = any("start" in seg and "end" in seg for seg in segments)
    if has_timestamps:
        segments.sort(key=lambda x: float(x.get("start", 0.0)))
    else:
        segments.sort(key=lambda x: int(x.get("text_index", 0)))

    norm_dir = tts_dir / "_norm"
    ensure_dir(norm_dir)

    with wave.open(str(out_wav), "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(config.audio.sample_rate)

        for idx, seg in enumerate(tqdm(segments, desc="Stitching", unit="segment")):
            seg_id = str(seg.get("group_id") or seg.get("segment_id"))
            in_path = tts_dir / f"{seg_id}.wav"
            if not in_path.exists():
                logger.warning("Missing TTS segment: %s", in_path)
                continue
            norm_path = norm_dir / f"{seg_id}.wav"
            _normalize_wav(in_path, norm_path, config.audio.sample_rate)

            with wave.open(str(norm_path), "rb") as wf_in:
                frames = wf_in.readframes(wf_in.getnframes())
                wf_out.writeframes(frames)

            pause_sec = _get_pause_after(segments, idx, config)
            if pause_sec > 0:
                silence_frames = int(pause_sec * config.audio.sample_rate)
                wf_out.writeframes(b"\x00\x00" * silence_frames)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        normalize_path(out_wav),
        normalize_path(out_mp3),
    ]
    run_command(cmd)
    return out_wav
