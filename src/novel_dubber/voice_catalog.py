import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import yaml

from .logging_utils import get_logger
from .utils import normalize_path


logger = get_logger(__name__)


@dataclass
class VoiceSample:
    sample_id: str
    audio: str
    gender: str
    ref_text: str
    name: str = ""
    locale: str = ""


def _load_yaml_ignoring_tags(path: Path) -> Dict[str, object]:
    class IgnoreTagsLoader(yaml.SafeLoader):
        pass

    def _construct_undefined(loader, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_scalar(node)

    IgnoreTagsLoader.add_constructor(None, _construct_undefined)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=IgnoreTagsLoader) or {}


def _normalize_gender(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("male", "m", "man", "boy"):
            return "male"
        if v in ("female", "f", "woman", "girl"):
            return "female"
        if v in ("unknown", "other", "na"):
            return "unknown"
        if v.isdigit():
            value = int(v)
        else:
            return "unknown"
    if isinstance(value, int):
        if value == 1:
            return "female"
        if value == 0:
            return "male"
    return "unknown"


def _extract_prompt_text(engines_path: Path) -> str:
    if not engines_path.exists():
        return ""
    data = _load_yaml_ignoring_tags(engines_path)
    engines = data.get("engines", []) if isinstance(data, dict) else []
    if not engines:
        return ""
    url = str(engines[0].get("url", ""))
    if not url:
        return ""
    qs = parse_qs(urlparse(url).query)
    prompt = qs.get("prompt_text", [""])[0]
    return str(prompt)


def _build_catalog_from_gpt_sovits(
    samples_dir: Path, config_path: Path, engines_path: Path
) -> Dict[str, object]:
    config = _load_yaml_ignoring_tags(config_path) if config_path.exists() else {}
    speakers = config.get("http_gpt", []) if isinstance(config, dict) else []
    by_code: Dict[str, Dict[str, object]] = {}
    for item in speakers:
        code = str(item.get("code", "")).strip()
        if not code:
            continue
        by_code[code] = item

    ref_text = _extract_prompt_text(engines_path)

    samples: List[Dict[str, object]] = []
    for wav in sorted(samples_dir.glob("*.wav")):
        match = re.match(r"(\\d+)_", wav.name)
        if not match:
            continue
        code = match.group(1)
        meta = by_code.get(code, {})
        samples.append(
            {
                "id": code,
                "name": str(meta.get("name", "")),
                "audio": str(wav),
                "gender": _normalize_gender(meta.get("gender", "unknown")),
                "locale": str(meta.get("locale", "")),
            }
        )

    return {"default_ref_text": ref_text, "samples": samples}


def _load_catalog_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_voice_catalog(catalog_path: Path) -> List[VoiceSample]:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Voice catalog not found: {catalog_path}")
    data = _load_catalog_yaml(catalog_path)
    default_text = str(data.get("default_ref_text", ""))
    samples: List[VoiceSample] = []
    for item in data.get("samples", []) if isinstance(data, dict) else []:
        sample_id = str(item.get("id", "")).strip()
        audio = str(item.get("audio", "")).strip()
        if not sample_id or not audio:
            continue
        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = (catalog_path.parent / audio_path).resolve()
        gender = _normalize_gender(item.get("gender", "unknown"))
        ref_text = str(item.get("ref_text", default_text)).strip()
        name = str(item.get("name", "")).strip()
        locale = str(item.get("locale", "")).strip()
        samples.append(
            VoiceSample(
                sample_id=sample_id,
                audio=normalize_path(audio_path),
                gender=gender,
                ref_text=ref_text,
                name=name,
                locale=locale,
            )
        )
    return samples


def ensure_voice_catalog(catalog_path: Path, voices_dir: Path) -> List[VoiceSample]:
    if catalog_path.exists():
        return load_voice_catalog(catalog_path)

    legacy_dir = voices_dir / "gpt-sovits"
    alt_dir = voices_dir / "source" / "gpt-sovits"
    source_dir = legacy_dir if legacy_dir.exists() else alt_dir
    config_path = source_dir / "config.yaml"
    engines_path = source_dir / "engines.yaml"
    samples_dir = voices_dir / "samples"
    if not samples_dir.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Missing voice samples or config at {samples_dir} / {config_path}"
        )

    catalog = _build_catalog_from_gpt_sovits(samples_dir, config_path, engines_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(catalog, f, allow_unicode=True, sort_keys=False)
    logger.info("Generated voice catalog at %s", catalog_path)
    return load_voice_catalog(catalog_path)
