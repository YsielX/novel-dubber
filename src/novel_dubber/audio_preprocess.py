import subprocess
from pathlib import Path
from typing import Dict, List

from .config import AppConfig
from .logging_utils import get_logger
from .utils import ensure_dir, normalize_path, run_command, write_json


logger = get_logger(__name__)


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        normalize_path(path),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe parse error: {proc.stdout}") from exc


def preprocess_audio(audio_path: Path, workdir: Path, config: AppConfig) -> Dict[str, Path]:
    ensure_dir(workdir)
    chunks_dir = workdir / "chunks"
    ensure_dir(chunks_dir)

    wav_path = workdir / "audio.wav"
    if not wav_path.exists():
        logger.info("Converting audio to wav: %s", wav_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            normalize_path(audio_path),
            "-ac",
            "1",
            "-ar",
            str(config.audio.sample_rate),
            normalize_path(wav_path),
        ]
        run_command(cmd)

    manifest_path = chunks_dir / "manifest.json"
    if not manifest_path.exists():
        logger.info("Chunking audio into %s sec segments", config.audio.chunk_sec)
        pattern = chunks_dir / "chunk_%04d.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            normalize_path(wav_path),
            "-f",
            "segment",
            "-segment_time",
            str(config.audio.chunk_sec),
            "-c",
            "pcm_s16le",
            normalize_path(pattern),
        ]
        run_command(cmd)

        chunks = sorted(chunks_dir.glob("chunk_*.wav"))
        manifest: List[Dict[str, float]] = []
        for idx, chunk in enumerate(chunks):
            duration = _ffprobe_duration(chunk)
            manifest.append(
                {
                    "chunk": str(chunk.name),
                    "start_sec": idx * config.audio.chunk_sec,
                    "duration_sec": duration,
                }
            )
        write_json(manifest_path, manifest)
    else:
        logger.info("Using existing chunk manifest: %s", manifest_path)

    return {"wav": wav_path, "chunks_dir": chunks_dir, "manifest": manifest_path}
