from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import AppConfig
from .logging_utils import get_logger
from .utils import ensure_dir, format_command, read_json, write_json


logger = get_logger(__name__)


def run_diarization(chunks_dir: Path, manifest_path: Path, workdir: Path, config: AppConfig) -> Path:
    ensure_dir(workdir)
    diar_path = workdir / "diarization.json"
    if diar_path.exists():
        logger.info("Using existing diarization: %s", diar_path)
        return diar_path

    manifest = read_json(manifest_path)
    chunk_results: List[Dict[str, float]] = []

    for entry in tqdm(manifest, desc="Diarization", unit="chunk"):
        chunk_file = chunks_dir / entry["chunk"]
        chunk_out = chunks_dir / f"{chunk_file.stem}_diar.json"
        if not chunk_out.exists():
            cmd = format_command(
                config.diarization.command_template,
                audio=str(chunk_file),
                out=str(chunk_out),
            )
            logger.info("Running diarization: %s", chunk_file.name)
            from .utils import run_command

            run_command(cmd)

        segments = read_json(chunk_out)
        for seg in segments:
            seg_start = float(seg["start"]) + float(entry["start_sec"])
            seg_end = float(seg["end"]) + float(entry["start_sec"])
            chunk_results.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": seg.get("speaker", "UNKNOWN"),
                    "overlap": bool(seg.get("overlap", False)),
                }
            )

    chunk_results.sort(key=lambda x: x["start"])
    write_json(diar_path, chunk_results)
    return diar_path
