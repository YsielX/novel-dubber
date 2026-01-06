import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import AppConfig
from .llm_client import call_llm_json
from .logging_utils import get_logger
from .utils import normalize_path, read_json, read_jsonl, write_json
from .voice_catalog import ensure_voice_catalog


logger = get_logger(__name__)


def _system_prompt_gender() -> str:
    return (
        "You are a strict JSON generator. Output ONLY valid JSON: "
        "{\"characters\":[{\"name\":\"...\",\"gender\":\"male\"|\"female\"|\"unknown\"}]}"
    )


def _user_prompt_gender(payload: List[Dict[str, object]]) -> str:
    blob = json.dumps(payload, ensure_ascii=True)
    return (
        "Infer gender for each character based on examples. "
        "Only use male/female/unknown. If unsure, return unknown. "
        "Do not guess from name alone. "
        "Examples may be in Japanese, Chinese, or Korean. "
        f"\nCharacters: {blob}"
    )


def _normalize_gender(value: str) -> str:
    v = value.strip().lower()
    if v in ("male", "m", "man", "boy"):
        return "male"
    if v in ("female", "f", "woman", "girl"):
        return "female"
    return "unknown"


def _collect_examples(
    segments: List[Dict[str, object]], max_examples: int
) -> Dict[str, List[str]]:
    examples: Dict[str, List[str]] = {}
    for seg in segments:
        character = str(seg.get("character", "")).strip()
        if character in ("", "UNKNOWN", "NARRATOR"):
            continue
        role = str(seg.get("role_type", "")).upper()
        if role != "DIALOGUE":
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        bucket = examples.setdefault(character, [])
        if len(bucket) >= max_examples:
            continue
        bucket.append(text[:200])

    if not examples:
        return examples

    # Fallback: use narration lines mentioning the character when dialogue is scarce.
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        for character, bucket in examples.items():
            if len(bucket) >= max_examples:
                continue
            if character in text and text not in bucket:
                bucket.append(text[:200])

    return examples


def _infer_genders(
    characters: List[str],
    examples: Dict[str, List[str]],
    config: AppConfig,
) -> Dict[str, str]:
    genders: Dict[str, str] = {}
    window = max(1, config.voice_assign.gender_window_size)
    for start in tqdm(range(0, len(characters), window), desc="Gender inference", unit="window"):
        chunk = characters[start : start + window]
        payload = []
        for name in chunk:
            payload.append(
                {
                    "name": name,
                    "examples": examples.get(name, [])[: config.voice_assign.max_examples_per_character],
                }
            )
        messages = [
            {"role": "system", "content": _system_prompt_gender()},
            {"role": "user", "content": _user_prompt_gender(payload)},
        ]
        llm_out = call_llm_json(config, messages)
        items = llm_out.get("characters", [])
        if not isinstance(items, list):
            continue
        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            gender = _normalize_gender(str(item.get("gender", "unknown")))
            genders[name] = gender
    return genders


def _load_existing_genders(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data = read_json(path)
    if isinstance(data, dict):
        return {str(k): _normalize_gender(str(v)) for k, v in data.items()}
    return {}


def infer_character_genders(
    labeled_segments_path: Path, out_path: Path, config: AppConfig
) -> Dict[str, str]:
    segments = read_jsonl(labeled_segments_path)
    characters = sorted(
        {
            str(seg.get("character", "")).strip()
            for seg in segments
            if str(seg.get("character", "")).strip() not in ("", "UNKNOWN", "NARRATOR")
        }
    )
    if not characters:
        write_json(out_path, {})
        return {}

    existing = _load_existing_genders(out_path)
    pending = [c for c in characters if c not in existing]
    if not pending:
        return existing

    examples = _collect_examples(segments, config.voice_assign.max_examples_per_character)
    inferred = _infer_genders(pending, examples, config)
    merged = dict(existing)
    merged.update(inferred)
    write_json(out_path, merged)
    return merged


def _pick_voice(
    rng: random.Random,
    candidates: List[Tuple[str, str, str]],
    used: set,
    allow_reuse: bool,
) -> Optional[Tuple[str, str, str]]:
    available = [c for c in candidates if c[0] not in used]
    if available:
        choice = rng.choice(available)
        used.add(choice[0])
        return choice
    if allow_reuse and candidates:
        return rng.choice(candidates)
    return None


def assign_voices_from_text(
    labeled_segments_path: Path,
    out_voice_map_path: Path,
    workdir: Path,
    config: AppConfig,
    voices_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    if out_voice_map_path.exists() and not force:
        logger.info("Using existing voice map: %s", out_voice_map_path)
        return out_voice_map_path

    voices_dir = voices_dir or Path("voices")
    catalog_path = Path(config.voice_assign.catalog_path)
    samples = ensure_voice_catalog(catalog_path, voices_dir)
    if not samples:
        raise RuntimeError(f"No voice samples found in {voices_dir}")

    genders_path = workdir / "character_genders.json"
    genders = infer_character_genders(labeled_segments_path, genders_path, config)

    segments = read_jsonl(labeled_segments_path)
    characters = sorted(
        {
            str(seg.get("character", "")).strip()
            for seg in segments
            if str(seg.get("character", "")).strip() not in ("", "UNKNOWN", "NARRATOR")
        }
    )

    by_gender: Dict[str, List[Tuple[str, str, str]]] = {"male": [], "female": [], "unknown": []}
    for sample in samples:
        ref_text = sample.ref_text
        by_gender[sample.gender].append((sample.sample_id, sample.audio, ref_text))

    all_candidates = [item for group in by_gender.values() for item in group]
    rng = random.Random(config.voice_assign.random_seed)
    used: set = set()

    voice_map: Dict[str, Dict[str, object]] = {}
    if "NARRATOR" not in voice_map:
        narrator_gender = _normalize_gender(config.voice_assign.narrator_gender)
        candidates = by_gender.get(narrator_gender) or all_candidates
        choice = _pick_voice(rng, candidates, used, config.voice_assign.allow_reuse)
        if choice:
            sample_id, audio, ref_text = choice
            voice_map["NARRATOR"] = {
                "refs": [
                    {
                        "audio": normalize_path(Path(audio)),
                        "text": ref_text,
                    }
                ],
                "gender": narrator_gender,
                "voice_id": sample_id,
            }

    for name in characters:
        desired = _normalize_gender(genders.get(name, "unknown"))
        candidates = by_gender.get(desired) or []
        if not candidates:
            candidates = all_candidates
        choice = _pick_voice(rng, candidates, used, config.voice_assign.allow_reuse)
        if not choice:
            logger.warning("No available voice sample for %s", name)
            continue
        sample_id, audio, ref_text = choice
        voice_map[name] = {
            "refs": [
                {
                    "audio": normalize_path(Path(audio)),
                    "text": ref_text,
                }
            ],
            "gender": desired,
            "voice_id": sample_id,
        }

    out_voice_map_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_voice_map_path, voice_map)
    return out_voice_map_path
