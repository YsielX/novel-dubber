import argparse
import json
import math
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


def _write_jsonl(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--word-timestamps", action="store_true")
    args = parser.parse_args()

    audio = args.audio
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device.lower().startswith("cuda") else "cpu"
    model = WhisperModel(args.model, device=device, compute_type=args.compute_type)

    segments, _info = model.transcribe(
        audio,
        language=args.language or None,
        vad_filter=True,
        word_timestamps=args.word_timestamps,
    )

    results = []
    for seg in segments:
        conf = 0.0
        if seg.avg_logprob is not None:
            conf = float(max(0.0, min(1.0, math.exp(seg.avg_logprob))))
        results.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "confidence": conf,
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": str(w.word).strip(),
                    }
                    for w in (seg.words or [])
                    if str(w.word).strip()
                ],
            }
        )

    _write_jsonl(out, results)


if __name__ == "__main__":
    main()
