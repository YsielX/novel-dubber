import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LLMConfig:
    endpoint: str
    api_key_env: str
    model: str
    timeout_sec: int = 120
    temperature: float = 0.2
    usage_log: str = ""
    max_retries: int = 2
    max_tokens: int = 0
    chat_log: str = ""


@dataclass
class ASRConfig:
    backend: str
    command_template: str
    device: str = "cpu"
    word_timestamps: bool = False


@dataclass
class DiarizationConfig:
    backend: str
    command_template: str
    overlap_threshold: float = 0.25
    enabled: bool = True


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_sec: int = 1800
    min_segment_sec: float = 1.0
    max_segment_sec: float = 30.0
    merge_pause_sec: float = 0.6
    max_pause_sec: float = 2.0


@dataclass
class LabelingConfig:
    mode: str = "per_segment"
    window_size: int = 12
    window_overlap: int = 4
    initial_characters: List[str] = field(default_factory=list)
    character_list_path: str = ""


@dataclass
class CharacterDiscoveryConfig:
    window_size: int = 80
    window_overlap: int = 0


@dataclass
class AlignmentConfig:
    search_window: int = 120
    max_combine: int = 30
    min_score: float = 0.55
    advance_on_miss: int = 0
    offset_window_segments: int = 20
    offset_search_words: int = 400
    use_kana: bool = True


@dataclass
class VoiceRefConfig:
    min_sec: float = 3.0
    max_sec: float = 20.0
    preferred_min_sec: float = 5.0
    preferred_max_sec: float = 15.0
    max_refs_per_character: int = 3


@dataclass
class TextDubConfig:
    default_pause_sec: float = 0.5
    punctuation_pause_sec: Dict[str, float] = field(default_factory=dict)
    quote_pause_sec: float = 0.2


@dataclass
class TranslationConfig:
    glossary_path: str = ""
    mode: str = "per_segment"
    window_size: int = 12
    window_overlap: int = 4
    enabled: bool = True


@dataclass
class TTSHttpConfig:
    endpoint: str
    json_body_template: Dict[str, str]


@dataclass
class TTSCLIConfig:
    command_template: str


@dataclass
class TTSConfig:
    mode: str = "http"
    http: Optional[TTSHttpConfig] = None
    cli: Optional[TTSCLIConfig] = None


@dataclass
class VoiceAssignConfig:
    catalog_path: str = "voices/catalog.yaml"
    random_seed: int = 13
    allow_reuse: bool = True
    narrator_gender: str = "unknown"
    max_examples_per_character: int = 6
    gender_window_size: int = 20


@dataclass
class AppConfig:
    llm: LLMConfig
    asr: ASRConfig
    diarization: DiarizationConfig
    audio: AudioConfig
    labeling: LabelingConfig
    character_discovery: CharacterDiscoveryConfig
    alignment: AlignmentConfig
    voice_refs: VoiceRefConfig
    text_dub: TextDubConfig
    translation: TranslationConfig
    tts: TTSConfig
    voice_assign: VoiceAssignConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    return data[key]


def load_config(path: Optional[Path] = None) -> AppConfig:
    if path is None:
        path = Path("config.yaml")
    data = _load_yaml(path)

    llm = _get(data, "llm", {})
    asr = _get(data, "asr", {})
    diar = _get(data, "diarization", {})
    audio = _get(data, "audio", {})
    labeling = _get(data, "labeling", {})
    char_disc = _get(data, "character_discovery", {})
    alignment = _get(data, "alignment", {})
    voice_refs = _get(data, "voice_refs", {})
    text_dub = _get(data, "text_dub", {})
    translation = _get(data, "translation", {})
    tts = _get(data, "tts", {})
    voice_assign = _get(data, "voice_assign", {})

    tts_http = _get(tts, "http")
    tts_cli = _get(tts, "cli")

    return AppConfig(
        llm=LLMConfig(
            endpoint=str(_get(llm, "endpoint", "")),
            api_key_env=str(_get(llm, "api_key_env", "OPENAI_API_KEY")),
            model=str(_get(llm, "model", "")),
            timeout_sec=int(_get(llm, "timeout_sec", 120)),
            temperature=float(_get(llm, "temperature", 0.2)),
            usage_log=str(_get(llm, "usage_log", "")),
            max_retries=int(_get(llm, "max_retries", 2)),
            max_tokens=int(_get(llm, "max_tokens", 0)),
            chat_log=str(_get(llm, "chat_log", "")),
        ),
        asr=ASRConfig(
            backend=str(_get(asr, "backend", "command")),
            command_template=str(_get(asr, "command_template", "")),
            device=str(_get(asr, "device", "cpu")),
            word_timestamps=bool(_get(asr, "word_timestamps", False)),
        ),
        diarization=DiarizationConfig(
            backend=str(_get(diar, "backend", "command")),
            command_template=str(_get(diar, "command_template", "")),
            overlap_threshold=float(_get(diar, "overlap_threshold", 0.25)),
            enabled=bool(_get(diar, "enabled", True)),
        ),
        audio=AudioConfig(
            sample_rate=int(_get(audio, "sample_rate", 16000)),
            chunk_sec=int(_get(audio, "chunk_sec", 1800)),
            min_segment_sec=float(_get(audio, "min_segment_sec", 1.0)),
            max_segment_sec=float(_get(audio, "max_segment_sec", 30.0)),
            merge_pause_sec=float(_get(audio, "merge_pause_sec", 0.6)),
            max_pause_sec=float(_get(audio, "max_pause_sec", 2.0)),
        ),
        labeling=LabelingConfig(
            mode=str(_get(labeling, "mode", "per_segment")),
            window_size=int(_get(labeling, "window_size", 12)),
            window_overlap=int(_get(labeling, "window_overlap", 4)),
            initial_characters=list(_get(labeling, "initial_characters", []) or []),
            character_list_path=str(_get(labeling, "character_list_path", "")),
        ),
        character_discovery=CharacterDiscoveryConfig(
            window_size=int(_get(char_disc, "window_size", 80)),
            window_overlap=int(_get(char_disc, "window_overlap", 0)),
        ),
        alignment=AlignmentConfig(
            search_window=int(_get(alignment, "search_window", 120)),
            max_combine=int(_get(alignment, "max_combine", 6)),
            min_score=float(_get(alignment, "min_score", 0.55)),
            advance_on_miss=int(_get(alignment, "advance_on_miss", 0)),
            offset_window_segments=int(_get(alignment, "offset_window_segments", 20)),
            offset_search_words=int(_get(alignment, "offset_search_words", 400)),
            use_kana=bool(_get(alignment, "use_kana", True)),
        ),
        voice_refs=VoiceRefConfig(
            min_sec=float(_get(voice_refs, "min_sec", 3.0)),
            max_sec=float(_get(voice_refs, "max_sec", 20.0)),
            preferred_min_sec=float(_get(voice_refs, "preferred_min_sec", 5.0)),
            preferred_max_sec=float(_get(voice_refs, "preferred_max_sec", 15.0)),
            max_refs_per_character=int(_get(voice_refs, "max_refs_per_character", 3)),
        ),
        text_dub=TextDubConfig(
            default_pause_sec=float(_get(text_dub, "default_pause_sec", 0.5)),
            punctuation_pause_sec=dict(_get(text_dub, "punctuation_pause_sec", {})),
            quote_pause_sec=float(_get(text_dub, "quote_pause_sec", 0.2)),
        ),
        translation=TranslationConfig(
            glossary_path=str(_get(translation, "glossary_path", "")),
            mode=str(_get(translation, "mode", "per_segment")),
            window_size=int(_get(translation, "window_size", 12)),
            window_overlap=int(_get(translation, "window_overlap", 4)),
            enabled=bool(_get(translation, "enabled", True)),
        ),
        tts=TTSConfig(
            mode=str(_get(tts, "mode", "http")),
            http=TTSHttpConfig(
                endpoint=str(_get(tts_http or {}, "endpoint", "")),
                json_body_template=dict(_get(tts_http or {}, "json_body_template", {})),
            )
            if tts_http
            else None,
            cli=TTSCLIConfig(command_template=str(_get(tts_cli or {}, "command_template", "")))
            if tts_cli
            else None,
        ),
        voice_assign=VoiceAssignConfig(
            catalog_path=str(_get(voice_assign, "catalog_path", "voices/catalog.yaml")),
            random_seed=int(_get(voice_assign, "random_seed", 13)),
            allow_reuse=bool(_get(voice_assign, "allow_reuse", True)),
            narrator_gender=str(_get(voice_assign, "narrator_gender", "unknown")),
            max_examples_per_character=int(_get(voice_assign, "max_examples_per_character", 6)),
            gender_window_size=int(_get(voice_assign, "gender_window_size", 20)),
        ),
    )


def load_api_key(config: AppConfig) -> str:
    key = os.getenv(config.llm.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"Missing API key in env var: {config.llm.api_key_env}"
        )
    return key
