from pathlib import Path
from typing import Optional

import typer

from .asr import run_asr
from .audio_preprocess import preprocess_audio
from .alignment import align_text_to_asr, merge_alignment
from .config import AppConfig, load_config
from .diarization import run_diarization
from .character_discovery import discover_characters
from .labeling import label_segments
from .logging_utils import setup_logging
from .merge import build_segments_from_asr, merge_asr_diarization
from .stitch import stitch_segments
from .text_mode import build_text_segments
from .translation import translate_segments
from .tts import synthesize_segments
from .voice_refs import extract_voice_refs


app = typer.Typer(add_completion=False)


def _load_config(config_path: Optional[Path]) -> AppConfig:
    return load_config(config_path)


def _set_usage_log(cfg: AppConfig, base_dir: Path) -> None:
    if not cfg.llm.usage_log:
        cfg.llm.usage_log = str(base_dir / "llm_usage.jsonl")


@app.callback()
def main(log_level: str = typer.Option("INFO", help="Log level")) -> None:
    setup_logging(log_level)


@app.command("audio-analyze")
def audio_analyze(
    audio: Path = typer.Option(..., help="Input audiobook audio file"),
    out: Path = typer.Option(..., help="Workdir for cached outputs"),
    language: Optional[str] = typer.Option(None, help="Language hint (e.g. ja)"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    results = preprocess_audio(audio, out, cfg)
    asr_path = run_asr(results["chunks_dir"], results["manifest"], out, cfg, language)
    if cfg.diarization.enabled:
        diar_path = run_diarization(results["chunks_dir"], results["manifest"], out, cfg)
        merge_asr_diarization(asr_path, diar_path, out / "segments.jsonl", cfg)
    else:
        build_segments_from_asr(asr_path, out / "segments.jsonl")


@app.command("audio-dump-voices")
def audio_dump_voices(
    workdir: Path = typer.Option(..., help="Workdir with segments.jsonl"),
    out: Path = typer.Option(..., help="Output workdir"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, out)
    labeled = label_segments(workdir / "segments.jsonl", out / "labeled_segments.jsonl", cfg)
    extract_voice_refs(labeled, workdir / "audio.wav", out, cfg)


@app.command("audio-discover-characters")
def audio_discover_characters(
    workdir: Path = typer.Option(..., help="Workdir with segments.jsonl"),
    out: Path = typer.Option(..., help="Output workdir"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, out)
    out.mkdir(parents=True, exist_ok=True)
    discover_characters(workdir / "segments.jsonl", out / "characters.json", cfg)


@app.command("text-analyze")
def text_analyze(
    text: Path = typer.Option(..., help="Input novel text file"),
    out: Path = typer.Option(..., help="Workdir for cached outputs"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    out.mkdir(parents=True, exist_ok=True)
    build_text_segments(text, out / "text_segments.jsonl", cfg)


@app.command("text-discover-characters")
def text_discover_characters(
    workdir: Path = typer.Option(..., help="Workdir with text_segments.jsonl"),
    out: Path = typer.Option(..., help="Output workdir"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, out)
    out.mkdir(parents=True, exist_ok=True)
    discover_characters(workdir / "text_segments.jsonl", out / "characters.json", cfg)


@app.command("text-label")
def text_label(
    workdir: Path = typer.Option(..., help="Workdir with text_segments.jsonl"),
    out: Path = typer.Option(..., help="Output workdir"),
    voice_map: Optional[Path] = typer.Option(None, help="Optional voice_map.json to constrain characters"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, out)
    out.mkdir(parents=True, exist_ok=True)
    char_path = voice_map if voice_map else None
    label_segments(
        workdir / "text_segments.jsonl",
        out / "labeled_segments.jsonl",
        cfg,
        character_list_path=char_path,
    )


@app.command("text-align")
def text_align(
    audio: Path = typer.Option(..., help="Input audiobook audio file"),
    workdir: Path = typer.Option(..., help="Workdir with text_segments.jsonl"),
    language: Optional[str] = typer.Option(None, help="Language hint (e.g. ja)"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    results = preprocess_audio(audio, workdir, cfg)
    asr_path = run_asr(results["chunks_dir"], results["manifest"], workdir, cfg, language)
    raw_path = workdir / "asr_raw.jsonl"
    align_src = raw_path if raw_path.exists() else asr_path
    align_text_to_asr(
        workdir / "text_segments.jsonl",
        align_src,
        workdir / "aligned_segments.jsonl",
        cfg,
    )


@app.command("text-dump-voices")
def text_dump_voices(
    workdir: Path = typer.Option(..., help="Workdir with labeled and aligned segments"),
    out: Path = typer.Option(..., help="Output workdir"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    out.mkdir(parents=True, exist_ok=True)
    merged = merge_alignment(
        workdir / "labeled_segments.jsonl",
        workdir / "aligned_segments.jsonl",
        out / "aligned_labeled_segments.jsonl",
    )
    extract_voice_refs(merged, workdir / "audio.wav", out, cfg)


@app.command("audio-dub")
def audio_dub(
    workdir: Path = typer.Option(..., help="Workdir with labeled segments and voice map"),
    target_lang: str = typer.Option(..., help="Target language code"),
    out: Path = typer.Option(..., help="Output directory"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, workdir)
    out.mkdir(parents=True, exist_ok=True)
    translated = translate_segments(
        workdir / "labeled_segments.jsonl",
        workdir / "translated_segments.jsonl",
        target_lang,
        cfg,
    )
    tts_dir = workdir / "tts_segments"
    synthesize_segments(translated, workdir / "voice_map.json", tts_dir, cfg, target_lang)
    stitch_segments(translated, tts_dir, out / "final_audio.wav", out / "final_audio.mp3", cfg)


@app.command("text-dub")
def text_dub(
    text: Path = typer.Option(..., help="Input novel text file"),
    voice_map: Path = typer.Option(..., help="Voice mapping JSON"),
    target_lang: str = typer.Option(..., help="Target language code"),
    out: Path = typer.Option(..., help="Output directory"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, out)
    out.mkdir(parents=True, exist_ok=True)
    segments = build_text_segments(text, out / "text_segments.jsonl", cfg)
    labeled = label_segments(
        segments,
        out / "labeled_segments.jsonl",
        cfg,
        character_list_path=voice_map,
    )
    translated = translate_segments(labeled, out / "translated_segments.jsonl", target_lang, cfg)
    tts_dir = out / "tts_segments"
    synthesize_segments(translated, voice_map, tts_dir, cfg, target_lang)
    stitch_segments(translated, tts_dir, out / "final_audio.wav", out / "final_audio.mp3", cfg)


@app.command("run-audio")
def run_audio(
    audio: Path = typer.Option(..., help="Input audiobook audio file"),
    workdir: Path = typer.Option(..., help="Workdir for cached outputs"),
    target_lang: str = typer.Option(..., help="Target language code"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    language: Optional[str] = typer.Option(None, help="Language hint (e.g. ja)"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, workdir)
    results = preprocess_audio(audio, workdir, cfg)
    asr_path = run_asr(results["chunks_dir"], results["manifest"], workdir, cfg, language)
    if cfg.diarization.enabled:
        diar_path = run_diarization(results["chunks_dir"], results["manifest"], workdir, cfg)
        merge_asr_diarization(asr_path, diar_path, workdir / "segments.jsonl", cfg)
    else:
        build_segments_from_asr(asr_path, workdir / "segments.jsonl")
    labeled = label_segments(workdir / "segments.jsonl", workdir / "labeled_segments.jsonl", cfg)
    extract_voice_refs(labeled, workdir / "audio.wav", workdir, cfg)
    translated = translate_segments(
        workdir / "labeled_segments.jsonl",
        workdir / "translated_segments.jsonl",
        target_lang,
        cfg,
    )
    tts_dir = workdir / "tts_segments"
    synthesize_segments(translated, workdir / "voice_map.json", tts_dir, cfg, target_lang)
    outdir = out or (workdir / "out")
    outdir.mkdir(parents=True, exist_ok=True)
    stitch_segments(translated, tts_dir, outdir / "final_audio.wav", outdir / "final_audio.mp3", cfg)


@app.command("run-text")
def run_text(
    text: Path = typer.Option(..., help="Input novel text file"),
    target_lang: str = typer.Option(..., help="Target language code"),
    workdir: Path = typer.Option(..., help="Workdir for cached outputs"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
    audio: Optional[Path] = typer.Option(None, help="Optional audio for alignment"),
    voice_map: Optional[Path] = typer.Option(None, help="Optional voice mapping JSON"),
    language: Optional[str] = typer.Option(None, help="Language hint (e.g. ja)"),
    config: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    cfg = _load_config(config)
    _set_usage_log(cfg, workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    segments = build_text_segments(text, workdir / "text_segments.jsonl", cfg)

    if audio:
        discover_characters(workdir / "text_segments.jsonl", workdir / "characters.json", cfg)
        labeled = label_segments(segments, workdir / "labeled_segments.jsonl", cfg)
        results = preprocess_audio(audio, workdir, cfg)
        asr_path = run_asr(results["chunks_dir"], results["manifest"], workdir, cfg, language)
        raw_path = workdir / "asr_raw.jsonl"
        align_src = raw_path if raw_path.exists() else asr_path
        align_text_to_asr(
            workdir / "text_segments.jsonl",
            align_src,
            workdir / "aligned_segments.jsonl",
            cfg,
        )
        merged = merge_alignment(
            workdir / "labeled_segments.jsonl",
            workdir / "aligned_segments.jsonl",
            workdir / "aligned_labeled_segments.jsonl",
        )
        extract_voice_refs(merged, workdir / "audio.wav", workdir, cfg)
        voice_map_path = workdir / "voice_map.json"
    else:
        if not voice_map:
            raise typer.BadParameter("Provide --voice-map when --audio is not set.")
        labeled = label_segments(
            segments,
            workdir / "labeled_segments.jsonl",
            cfg,
            character_list_path=voice_map,
        )
        voice_map_path = voice_map

    translated = translate_segments(
        labeled,
        workdir / "translated_segments.jsonl",
        target_lang,
        cfg,
    )
    tts_dir = workdir / "tts_segments"
    synthesize_segments(translated, voice_map_path, tts_dir, cfg, target_lang)
    outdir = out or (workdir / "out")
    outdir.mkdir(parents=True, exist_ok=True)
    stitch_segments(translated, tts_dir, outdir / "final_audio.wav", outdir / "final_audio.mp3", cfg)


if __name__ == "__main__":
    app()
