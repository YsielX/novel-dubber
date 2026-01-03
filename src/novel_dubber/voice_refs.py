import audioop
import json
import wave
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import AppConfig
from .logging_utils import get_logger
from .utils import ensure_dir, normalize_path, read_jsonl, write_json


logger = get_logger(__name__)


def _segment_rms(wav_path: Path, start_sec: float, end_sec: float) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frame_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        start_frame = int(start_sec * frame_rate)
        end_frame = int(end_sec * frame_rate)
        wf.setpos(max(0, start_frame))
        frames = wf.readframes(max(0, end_frame - start_frame))
        if not frames:
            return 0.0
        return float(audioop.rms(frames, sample_width))


def _extract_wav_segment(wav_path: Path, start_sec: float, end_sec: float, out_path: Path) -> None:
    from .utils import run_command

    duration = max(0.1, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-i",
        normalize_path(wav_path),
        "-ac",
        "1",
        normalize_path(out_path),
    ]
    run_command(cmd)


def extract_voice_refs(
    labeled_path: Path,
    wav_path: Path,
    workdir: Path,
    config: AppConfig,
) -> Path:
    voice_map_path = workdir / "voice_map.json"
    if voice_map_path.exists():
        logger.info("Using existing voice map: %s", voice_map_path)
        return voice_map_path

    segments = read_jsonl(labeled_path)
    ensure_dir(workdir / "voice_map")

    candidates_by_char: Dict[str, List[Dict[str, object]]] = {}
    for seg in segments:
        character = str(seg.get("character", "UNKNOWN"))
        if character == "UNKNOWN":
            continue
        if bool(seg.get("overlap", False)):
            continue
        if "start" not in seg or "end" not in seg:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = end - start
        if dur < config.voice_refs.min_sec or dur > config.voice_refs.max_sec:
            continue
        candidates_by_char.setdefault(character, []).append(seg)

    voice_map: Dict[str, Dict[str, object]] = {}
    report_path = workdir / "voice_refs_report.jsonl"
    if report_path.exists():
        report_path.unlink()

    for character, segs in candidates_by_char.items():
        scored: List[Dict[str, object]] = []
        for seg in tqdm(segs, desc=f"Voice refs {character}", unit="segment"):
            start = float(seg["start"])
            end = float(seg["end"])
            dur = end - start
            rms = _segment_rms(wav_path, start, end)
            score = 0.0
            if config.voice_refs.preferred_min_sec <= dur <= config.voice_refs.preferred_max_sec:
                score += 1.0
            score += min(1.0, rms / 10000.0)
            scored.append({"seg": seg, "score": score, "rms": rms, "dur": dur})

        scored.sort(key=lambda x: x["score"], reverse=True)
        selected = scored[: config.voice_refs.max_refs_per_character]

        refs: List[Dict[str, str]] = []
        char_dir = workdir / "voice_map" / character
        ensure_dir(char_dir)
        for i, item in enumerate(selected, start=1):
            seg = item["seg"]
            out_wav = char_dir / f"ref_{i:03d}.wav"
            _extract_wav_segment(wav_path, float(seg["start"]), float(seg["end"]), out_wav)
            refs.append({"audio": str(out_wav), "text": str(seg.get("text", ""))})

            with report_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "character": character,
                            "segment_id": seg.get("segment_id"),
                            "audio": str(out_wav),
                            "text": str(seg.get("text", "")),
                            "rms": item["rms"],
                            "duration": item["dur"],
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        clusters = sorted(
            {
                str(seg.get("speaker_cluster", "UNKNOWN"))
                for seg in segs
                if str(seg.get("speaker_cluster", "UNKNOWN")) != "UNKNOWN"
            }
        )

        voice_map[character] = {"refs": refs, "clusters": clusters}

    write_json(voice_map_path, voice_map)
    return voice_map_path
