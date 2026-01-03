import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering


def _write_json(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=True, indent=2)


def _read_pcm(wav_path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(wav_path))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float32)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    return wav, sr


def _frame_generator(pcm: bytes, frame_ms: int, sample_rate: int) -> List[bytes]:
    frame_len = int(sample_rate * frame_ms / 1000) * 2
    frames = []
    for i in range(0, len(pcm), frame_len):
        chunk = pcm[i : i + frame_len]
        if len(chunk) < frame_len:
            break
        frames.append(chunk)
    return frames


def _vad_segments(
    wav: np.ndarray, sample_rate: int, vad_level: int = 2, min_speech_sec: float = 0.3
) -> List[Tuple[float, float]]:
    vad = webrtcvad.Vad(vad_level)
    pcm16 = (wav * 32767).astype(np.int16).tobytes()
    frame_ms = 30
    frames = _frame_generator(pcm16, frame_ms, sample_rate)
    timestamps = []
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame, sample_rate)
        start = (i * frame_ms) / 1000.0
        end = ((i + 1) * frame_ms) / 1000.0
        timestamps.append((start, end, is_speech))

    segments: List[Tuple[float, float]] = []
    cur_start = None
    for start, end, is_speech in timestamps:
        if is_speech and cur_start is None:
            cur_start = start
        if not is_speech and cur_start is not None:
            if end - cur_start >= min_speech_sec:
                segments.append((cur_start, end))
            cur_start = None
    if cur_start is not None:
        segments.append((cur_start, timestamps[-1][1]))
    return segments


def _merge_short_gaps(segments: List[Tuple[float, float]], max_gap: float = 0.3) -> List[Tuple[float, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def _cluster_embeddings(
    embeddings: np.ndarray, num_speakers: Optional[int], distance_threshold: float
) -> List[int]:
    if len(embeddings) == 1:
        return [0]
    clustering = AgglomerativeClustering(
        n_clusters=num_speakers,
        distance_threshold=None if num_speakers else distance_threshold,
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return labels.tolist()


def _window_segments(
    segments: List[Tuple[float, float]],
    window_sec: float,
    step_sec: float,
    min_window_sec: float,
) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    for start, end in segments:
        if end - start < min_window_sec:
            continue
        if end - start <= window_sec:
            windows.append((start, end))
            continue
        cur = start
        while cur + min_window_sec <= end:
            win_end = min(cur + window_sec, end)
            if win_end - cur >= min_window_sec:
                windows.append((cur, win_end))
            cur += step_sec
    return windows


def _merge_labeled_windows(
    windows: List[Tuple[float, float]], labels: List[int], max_gap_sec: float
) -> List[Tuple[float, float, int]]:
    items = sorted(zip(windows, labels), key=lambda x: x[0][0])
    merged: List[Tuple[float, float, int]] = []
    cur_start = None
    cur_end = None
    cur_label = None
    for (start, end), label in items:
        if cur_label is None:
            cur_start, cur_end, cur_label = start, end, label
            continue
        if label == cur_label and start - float(cur_end) <= max_gap_sec:
            cur_end = max(float(cur_end), end)
        else:
            merged.append((float(cur_start), float(cur_end), int(cur_label)))
            cur_start, cur_end, cur_label = start, end, label
    if cur_label is not None:
        merged.append((float(cur_start), float(cur_end), int(cur_label)))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vad-level", type=int, default=3)
    parser.add_argument("--min-speech-sec", type=float, default=0.3)
    parser.add_argument("--max-gap-sec", type=float, default=0.3)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--min-window-sec", type=float, default=0.8)
    parser.add_argument("--distance-threshold", type=float, default=0.75)
    parser.add_argument("--num-speakers", type=int, default=0)
    args = parser.parse_args()

    wav_path = Path(args.audio)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav, sr = _read_pcm(wav_path)
    segments = _vad_segments(wav, sr, vad_level=args.vad_level, min_speech_sec=args.min_speech_sec)
    segments = _merge_short_gaps(segments, max_gap=args.max_gap_sec)
    windows = _window_segments(segments, args.window_sec, args.step_sec, args.min_window_sec)

    device = "cuda" if args.device.lower().startswith("cuda") else "cpu"
    encoder = VoiceEncoder(device=device)
    embeddings = []
    for start, end in windows:
        seg = wav[int(start * sr) : int(end * sr)]
        if len(seg) < int(args.min_window_sec * sr):
            continue
        emb = encoder.embed_utterance(seg)
        embeddings.append(emb)

    if not embeddings:
        _write_json(out_path, [])
        return

    embeddings_np = np.vstack(embeddings)
    num_speakers = args.num_speakers if args.num_speakers > 0 else None
    labels = _cluster_embeddings(embeddings_np, num_speakers, args.distance_threshold)

    diar = []
    merged = _merge_labeled_windows(windows, labels, max_gap_sec=args.step_sec * 1.1)
    for start, end, label in merged:
        diar.append(
            {
                "start": float(start),
                "end": float(end),
                "speaker": f"SPEAKER_{label:02d}",
                "overlap": False,
            }
        )
    _write_json(out_path, diar)


if __name__ == "__main__":
    main()
