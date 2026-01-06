#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

from novel_dubber.config import load_config
from novel_dubber.tts import (
    _cli_tts,
    _has_spoken_content,
    _http_tts,
    _normalize_voice_map,
    _prepare_ref_audio,
    _sanitize_tts_text,
    _select_ref,
)
from novel_dubber.utils import normalize_path, read_json


def _load_text(text: Optional[str], text_file: Optional[Path]) -> str:
    if text:
        return text
    if text_file:
        return text_file.read_text(encoding="utf-8").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a single line for a character using voice_map.json."
    )
    parser.add_argument("--voice-map", required=True, type=Path, help="Path to voice_map.json")
    parser.add_argument("--character", required=True, help="Character name in voice_map.json")
    parser.add_argument("--text", help="Text to speak")
    parser.add_argument("--text-file", type=Path, help="Text file containing the line to speak")
    parser.add_argument("--out", required=True, type=Path, help="Output wav path")
    parser.add_argument("--target-lang", required=True, help="Target language code for TTS")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()

    text = _load_text(args.text, args.text_file)
    if not text:
        raise SystemExit("Missing --text or --text-file content.")

    cfg = load_config(args.config)
    voice_map = _normalize_voice_map(read_json(args.voice_map))
    refs = voice_map.get(args.character) or voice_map.get("NARRATOR")
    if not refs:
        raise SystemExit(f"No refs for character: {args.character}")

    ref_audio, ref_text = _select_ref(refs, args.voice_map.parent / "tts_ref_cache")
    if not ref_audio:
        raise SystemExit(f"No usable ref audio for character: {args.character}")
    print(f"Using ref audio: {ref_audio}")

    ref_audio = _prepare_ref_audio(ref_audio, args.voice_map.parent / "tts_ref_cache")
    ref_audio = normalize_path(Path(ref_audio))
    if not Path(ref_audio).exists():
        raise SystemExit(f"Ref audio not found: {ref_audio}")

    text = _sanitize_tts_text(text)
    if not text or not _has_spoken_content(text):
        raise SystemExit("Text has no spoken content.")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.tts.mode == "http":
        _http_tts(cfg, text, ref_audio, ref_text, args.target_lang, out_path)
    else:
        _cli_tts(cfg, text, ref_audio, ref_text, args.target_lang, out_path)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
