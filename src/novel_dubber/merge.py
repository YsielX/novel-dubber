from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import AppConfig
from .logging_utils import get_logger
from .utils import append_jsonl, read_json, read_jsonl


logger = get_logger(__name__)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_asr_diarization(
    asr_path: Path, diar_path: Path, out_path: Path, config: AppConfig
) -> Path:
    if out_path.exists():
        logger.info("Using existing merged segments: %s", out_path)
        return out_path

    asr_segments = read_jsonl(asr_path)
    diar_segments = read_json(diar_path)

    diar_segments.sort(key=lambda x: x["start"])

    merged: List[Dict[str, object]] = []
    for idx, seg in enumerate(tqdm(asr_segments, desc="Merging", unit="segment")):
        start = float(seg["start"])
        end = float(seg["end"])
        seg_len = max(0.001, end - start)

        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        overlap_dur = 0.0

        for diar in diar_segments:
            if diar["end"] < start:
                continue
            if diar["start"] > end:
                break
            ov = _overlap(start, end, float(diar["start"]), float(diar["end"]))
            if ov <= 0:
                continue
            if diar.get("overlap", False):
                overlap_dur += ov
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = diar.get("speaker", "UNKNOWN")

        if best_overlap / seg_len < 0.1:
            best_speaker = "UNKNOWN"
        if overlap_dur / seg_len >= config.diarization.overlap_threshold:
            best_speaker = "UNKNOWN"

        merged.append(
            {
                "segment_id": f"seg_{idx:06d}",
                "start": start,
                "end": end,
                "text": seg.get("text", ""),
                "confidence": float(seg.get("confidence", 0.0)),
                "speaker_cluster": best_speaker,
                "overlap": overlap_dur / seg_len >= config.diarization.overlap_threshold,
            }
        )

    append_jsonl(out_path, merged)
    return out_path


def build_segments_from_asr(asr_path: Path, out_path: Path) -> Path:
    if out_path.exists():
        logger.info("Using existing merged segments: %s", out_path)
        return out_path

    asr_segments = read_jsonl(asr_path)
    merged: List[Dict[str, object]] = []
    for idx, seg in enumerate(tqdm(asr_segments, desc="Segmenting", unit="segment")):
        merged.append(
            {
                "segment_id": f"seg_{idx:06d}",
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
                "confidence": float(seg.get("confidence", 0.0)),
                "speaker_cluster": "UNKNOWN",
                "overlap": False,
            }
        )

    append_jsonl(out_path, merged)
    return out_path
