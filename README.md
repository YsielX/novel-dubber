# novel-dubber

Production-grade CLI to dub audiobooks using a text-based pipeline with optional audio alignment, local ASR/diarization, an external OpenAI-compatible LLM for labeling/translation, and GPT-SoVITS for synthesis.

## Install

```bash
pip install -e .
```

## Quickstart

### Text-based (recommended)

Text only (manual voice map):

```bash
novel_dubber text-dub --text novel.txt --voice-map voice_map.json --target-lang en --out outdir
```

Text only (auto voice map from samples):

```bash
novel_dubber text-analyze --text novel.txt --out workdir
novel_dubber text-discover-characters --workdir workdir --out workdir
novel_dubber text-label --workdir workdir --out workdir
novel_dubber text-assign-voices --workdir workdir --out workdir
novel_dubber audio-dub --workdir workdir --target-lang en --out outdir
```

Text only, no translation (use source language for TTS):

```bash
novel_dubber run-text --text novel.txt --target-lang ja --workdir workdir --no-translate
```

You can also add `--no-translate` to `text-dub` or `audio-dub`.

Text + audio alignment (auto voice map):

```bash
novel_dubber run-text --text novel.txt --audio audiobook.mp3 --workdir workdir --target-lang en
```

Alignment finds the best text offset to match the audio start, then aligns with word-level timestamps.

If you omit `--audio`, `run-text` will auto-assign voices from `voices/catalog.yaml`.

Voice map schema:

```json
{
  "NARRATOR": {"ref_audio": "path/to.wav", "ref_text": "reference transcript"},
  "CharacterA": {"ref_audio": "path/to.wav", "ref_text": "reference transcript"}
}
```

Auto-generated voice map (from aligned audio):

```json
{
  "CharacterA": {
    "refs": [{"audio": "workdir/voice_map/CharacterA/ref_001.wav", "text": "..."}],
    "clusters": ["SPEAKER_00"]
  }
}
```

Auto-generated voice map (no audio, from samples):

```json
{
  "CharacterA": {
    "refs": [{"audio": "/abs/path/to/voices/samples/12_oral5.wav", "text": "..."}],
    "gender": "female",
    "voice_id": "12"
  }
}
```

## Workdir layout (cache/resume)

- `chunks/` audio chunks + `manifest.json`
- `diarization.json` combined diarization
- `asr_raw.jsonl` raw ASR with timestamps
- `asr_segments.jsonl` merged/split ASR segments
- `asr_words.jsonl` word-level timestamps (for alignment)
- `segments.jsonl` merged ASR + diarization (audio-only flow)
- `text_segments.jsonl` text-based segments
- `characters.json` discovered character list (text-based)
- `aligned_segments.jsonl` text segments aligned to ASR (optional)
- `labeled_segments.jsonl` role/character labeling (text-based)
- `aligned_labeled_segments.jsonl` labeled + alignment merged (for voice refs)
- `voice_map.json` + `voice_map/<character>/ref_###.wav`
- `translated_segments.jsonl`
- `tts_segments/group_*.wav`
- `tts_segments/tts_groups.jsonl`
- `final_audio.wav` + `final_audio.mp3`

## Config

Edit `config.yaml` or pass `--config path/to.yaml`.

Key settings:
- `llm.endpoint`, `llm.model`, `llm.api_key_env`
- `asr.command_template`, `asr.word_timestamps`, `diarization.command_template`
- `tts.mode` (`http` or `cli`) and adapter config
- `character_discovery.window_size`, `labeling.character_list_path`
- `alignment.search_window`, `alignment.max_combine`, `alignment.min_score`, `alignment.use_kana`, `alignment.offset_window_segments`
- `voice_assign.catalog_path`, `voice_assign.gender_window_size`
- `translation.enabled` (set false to skip LLM translation)

The LLM API key is read from the environment variable specified in `llm.api_key_env`.

## Full pipeline commands

Text-based with audio alignment (recommended step order):

```bash
# 1) split text into sentence-level segments
novel_dubber text-analyze --text novel.txt --out workdir

# 2) discover character list (LLM merges aliases across languages)
novel_dubber text-discover-characters --workdir workdir --out workdir

# 3) label each segment with role + character
novel_dubber text-label --workdir workdir --out workdir

# 4) align text segments to audio timestamps (optional)
novel_dubber text-align --audio audiobook.mp3 --workdir workdir --language ja

# 5) extract voice references from aligned audio
novel_dubber text-dump-voices --workdir workdir --out workdir

# 6) translate + TTS + stitch in text order
novel_dubber audio-dub --workdir workdir --target-lang en --out outdir
```

Text-only (manual voice map):

```bash
novel_dubber text-dub --text novel.txt --voice-map voice_map.json --target-lang en --out outdir
```

Text-only (auto voice map from samples):

```bash
novel_dubber text-analyze --text novel.txt --out workdir
novel_dubber text-discover-characters --workdir workdir --out workdir
novel_dubber text-label --workdir workdir --out workdir
novel_dubber text-assign-voices --workdir workdir --out workdir
novel_dubber audio-dub --workdir workdir --target-lang en --out outdir
```

Audio-only (legacy flow):

```bash
novel_dubber run-audio --audio audiobook.mp3 --workdir workdir --target-lang en
```

## External tool expectations

### ASR
The ASR command must output JSONL at the given `{out}` path. Each line:

```json
{"start": 12.34, "end": 15.67, "text": "...", "confidence": 0.91}
```

If `asr.word_timestamps` is enabled, include a `words` array:

```json
{
  "start": 12.34,
  "end": 15.67,
  "text": "...",
  "confidence": 0.91,
  "words": [{"start": 12.34, "end": 12.90, "word": "..." }]
}
```

Times must be in seconds relative to the input chunk audio.

### GPT-SoVITS
- **HTTP mode** expects a WAV response or JSON containing `audio_path` or `audio_base64`.
- **CLI mode** uses `tts.cli.command_template` with placeholders `{text_file}`, `{ref_audio}`, `{ref_text}`, `{out}`, `{target_lang}`.
- **TTS grouping**: consecutive segments with the same character are merged into a single TTS request. Group metadata is saved to `tts_segments/tts_groups.jsonl` and used by stitching.

## Voice catalog (no audiobook)

`voices/catalog.yaml` lists available reference samples and metadata. Example:

```yaml
default_ref_text: "..."
samples:
  - id: "12"
    name: "n12"
    audio: "samples/12_oral5.wav"
    gender: "female"
    locale: "zh-CN"
```

`text-assign-voices` will infer character genders from labeled text and randomly assign
matching voices. If the catalog is missing, it will be generated from
`voices/source/gpt-sovits/` (or legacy `voices/gpt-sovits/`) plus `voices/samples/`.

## Smoke tests

```bash
pytest -q
```
