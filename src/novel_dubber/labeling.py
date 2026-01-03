import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from .config import AppConfig
from .llm_client import call_llm_json
from .logging_utils import get_logger
from .utils import append_jsonl, read_json, read_jsonl


logger = get_logger(__name__)


def _system_prompt() -> str:
    return (
        "You are a strict JSON generator. Output ONLY valid JSON with keys: "
        "role_type, character. "
        "role_type must be NARRATION or DIALOGUE. "
        "character must be NARRATOR or a character name from the allowed list."
    )


def _system_prompt_window() -> str:
    return (
        "You are a strict JSON generator. Output ONLY valid JSON: "
        "{\"segments\":[{\"segment_id\":\"...\",\"role_type\":\"NARRATION\"|\"DIALOGUE\","
        "\"character\":\"NARRATOR\"|\"<name>\"}]}"
    )


def _user_prompt(
    segment_text: str,
    role_hint: str,
    known_characters: List[str],
    context: List[Dict[str, str]],
) -> str:
    known = ", ".join(known_characters) if known_characters else "NONE"
    context_json = json.dumps(context, ensure_ascii=True)
    return (
        "Label the current segment with role_type and character. "
        "Use the role_hint if it is provided, unless it is clearly wrong. "
        "Choose character names only from the allowed list. "
        "Do not create new names."
        f"\nAllowed characters: {known}"
        f"\nRole hint: {role_hint}"
        f"\nCurrent text: {segment_text}"
        f"\nRecent context: {context_json}"
        "\nReturn JSON: {"
        "\"role_type\":\"NARRATION\"|\"DIALOGUE\","
        "\"character\":\"NARRATOR\"|\"<name>\"}"
    )


def _user_prompt_window(
    segments: List[Dict[str, str]],
    known_characters: List[str],
    recent_context: List[Dict[str, str]],
) -> str:
    known = ", ".join(known_characters) if known_characters else "NONE"
    segments_json = json.dumps(segments, ensure_ascii=True)
    context_json = json.dumps(recent_context, ensure_ascii=True)
    return (
        "You will label consecutive text segments from a novel. "
        "For each segment, decide: role_type (NARRATION or DIALOGUE), "
        "character (NARRATOR or a character name). "
        "Use role_hint if provided (role_hint may be UNKNOWN), unless it is clearly wrong. "
        "Use consistent character names across segments. Choose from the allowed "
        "character list. Do not introduce new names. "
        "Do not output placeholder speaker IDs. "
        "Narration must use character NARRATOR."
        f"\nAllowed characters: {known}"
        f"\nRecent labeled context: {context_json}"
        f"\nSegments to label: {segments_json}"
        "\nReturn JSON: {"
        "\"segments\":["
        "{\"segment_id\":\"...\",\"role_type\":\"NARRATION\"|\"DIALOGUE\","
        "\"character\":\"NARRATOR\"|\"<name>\"}"
        "]}"
    )


def _normalize_role_character(role_type: str, character: str) -> Dict[str, str]:
    role = role_type.upper()
    char = character
    if role == "NARRATION" and char not in ("NARRATOR", "UNKNOWN"):
        char = "NARRATOR"
    if role == "DIALOGUE" and char == "NARRATOR":
        role = "NARRATION"
    if role not in ("NARRATION", "DIALOGUE"):
        role = "NARRATION"
    return {"role_type": role, "character": char}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def _load_character_list(
    segments_path: Path, config: AppConfig, override_path: Optional[Path] = None
) -> Tuple[Set[str], Dict[str, str]]:
    path = config.labeling.character_list_path
    if override_path:
        path = str(override_path)
    if not path:
        path = str(segments_path.parent / "characters.json")
    char_path = Path(path)
    if not char_path.exists():
        return set(), {}
    try:
        data = read_json(char_path)
    except Exception:
        return set(), {}
    allowed: Set[str] = set()
    alias_map: Dict[str, str] = {}

    if isinstance(data, dict) and "characters" in data:
        items = data.get("characters", [])
        for item in items:
            canonical = str(item.get("canonical", "")).strip()
            if not canonical:
                continue
            allowed.add(canonical)
            alias_map[_normalize_name(canonical)] = canonical
            for alias in item.get("aliases", []):
                alias = str(alias).strip()
                if not alias:
                    continue
                allowed.add(alias)
                alias_map[_normalize_name(alias)] = canonical
        return allowed, alias_map

    if isinstance(data, dict):
        for name in data.keys():
            name = str(name).strip()
            if not name:
                continue
            allowed.add(name)
            alias_map[_normalize_name(name)] = name
        return allowed, alias_map

    return set(), {}


def _map_character(alias_map: Dict[str, str], allowed: Set[str], name: str) -> str:
    key = _normalize_name(name)
    if key in alias_map:
        return alias_map[key]
    if allowed:
        for candidate in allowed:
            ckey = _normalize_name(candidate)
            if not ckey:
                continue
            if key and (key in ckey or ckey in key):
                return candidate
    return name




def _label_segments_per_segment(
    segments_path: Path,
    out_path: Path,
    config: AppConfig,
    character_list_path: Optional[Path] = None,
) -> Path:
    segments = read_jsonl(segments_path)
    labeled_existing = read_jsonl(out_path) if out_path.exists() else []
    processed_ids = {seg.get("segment_id") for seg in labeled_existing}
    results: List[Dict[str, object]] = []
    allowed_characters, alias_map = _load_character_list(
        segments_path, config, override_path=character_list_path
    )
    known_characters: Set[str] = set(config.labeling.initial_characters)
    known_characters.update(allowed_characters)

    for idx, seg in enumerate(tqdm(segments, desc="Labeling", unit="segment")):
        seg_id = str(seg["segment_id"])
        if seg_id in processed_ids:
            continue
        context_items: List[Dict[str, str]] = []
        for offset in range(-3, 4):
            if offset == 0:
                continue
            j = idx + offset
            if j < 0 or j >= len(segments):
                continue
            neighbor = segments[j]
            context_items.append(
                {
                    "text": str(neighbor.get("text", "")),
                    "role_type": str(neighbor.get("role_type", "")),
                    "character": str(neighbor.get("character", "")),
                }
            )

        messages = [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": _user_prompt(
                    str(seg.get("text", "")),
                    str(seg.get("role_type") or "UNKNOWN"),
                    sorted(known_characters),
                    context_items,
                ),
            },
        ]

        llm_out = call_llm_json(config, messages)
        role_type = str(llm_out.get("role_type", "NARRATION")).upper()
        character = str(llm_out.get("character", "")).strip()
        if not character:
            character = "UNKNOWN"
        if character.upper().startswith("SPEAKER"):
            character = "UNKNOWN"
        if allowed_characters:
            character = _map_character(alias_map, allowed_characters, character)
            if character not in allowed_characters and character != "NARRATOR":
                character = "UNKNOWN"
        norm = _normalize_role_character(role_type, character)

        out = dict(seg)
        out.update(
            {
                "role_type": norm["role_type"],
                "character": norm["character"],
            }
        )
        results.append(out)
        if out["character"] not in ("UNKNOWN", "NARRATOR", ""):
            known_characters.add(out["character"])

    if results:
        append_jsonl(out_path, results)
    else:
        if not out_path.exists():
            out_path.touch()
    return out_path


def _label_segments_windowed(
    segments_path: Path,
    out_path: Path,
    config: AppConfig,
    character_list_path: Optional[Path] = None,
) -> Path:
    segments = read_jsonl(segments_path)
    labeled_existing = read_jsonl(out_path) if out_path.exists() else []
    labeled_map = {str(seg.get("segment_id")): seg for seg in labeled_existing}

    allowed_characters, alias_map = _load_character_list(
        segments_path, config, override_path=character_list_path
    )
    known_characters: Set[str] = set(config.labeling.initial_characters)
    known_characters.update(allowed_characters)
    for seg in labeled_existing:
        character = str(seg.get("character"))
        if character not in ("UNKNOWN", "NARRATOR", ""):
            known_characters.add(character)

    if not segments:
        if not out_path.exists():
            out_path.touch()
        return out_path

    step = max(1, config.labeling.window_size - config.labeling.window_overlap)
    segment_index = {str(seg.get("segment_id")): seg for seg in segments}

    for start in tqdm(
        range(0, len(segments), step), desc="Labeling", unit="window"
    ):
        window = segments[start : start + config.labeling.window_size]
        if not window:
            continue

        recent_context: List[Dict[str, str]] = []
        ctx_start = max(0, start - config.labeling.window_overlap)
        for seg in segments[ctx_start:start]:
            seg_id = str(seg.get("segment_id"))
            labeled = labeled_map.get(seg_id)
            if not labeled:
                continue
            recent_context.append(
                {
                    "segment_id": seg_id,
                    "text": str(labeled.get("text", "")),
                    "role_type": str(labeled.get("role_type", "")),
                    "character": str(labeled.get("character", "")),
                }
            )

        payload_segments = [
            {
                "segment_id": str(seg.get("segment_id")),
                "text": str(seg.get("text", "")),
                "role_hint": str(seg.get("role_type") or "UNKNOWN"),
            }
            for seg in window
        ]

        messages = [
            {"role": "system", "content": _system_prompt_window()},
            {
                "role": "user",
                "content": _user_prompt_window(
                    payload_segments,
                    sorted(known_characters),
                    recent_context,
                ),
            },
        ]

        llm_out = call_llm_json(config, messages)
        items = llm_out.get("segments", [])
        if not isinstance(items, list):
            items = []

        for item in items:
            seg_id = str(item.get("segment_id", ""))
            if not seg_id:
                continue
            base = segment_index.get(seg_id)
            if not base:
                continue

            role_type = str(item.get("role_type", "NARRATION")).upper()
            character = str(item.get("character", "")).strip() or "UNKNOWN"
            if character.upper().startswith("SPEAKER"):
                character = "UNKNOWN"
            canonical = _map_character(alias_map, allowed_characters, character)
            norm = _normalize_role_character(role_type, canonical)
            existing = labeled_map.get(seg_id)
            existing_char = ""
            existing_role = ""
            if existing is not None:
                existing_char = str(existing.get("character", ""))
                existing_role = str(existing.get("role_type", ""))
                if existing_char not in ("", "UNKNOWN"):
                    if not (
                        existing_char == "NARRATOR"
                        and norm["character"] != "NARRATOR"
                        and norm["role_type"] == "DIALOGUE"
                    ):
                        continue

            if existing is None or existing_char in ("", "UNKNOWN") or existing_role != norm["role_type"]:
                out = dict(base)
                out.update(
                    {
                        "role_type": norm["role_type"],
                        "character": norm["character"],
                    }
                )
                labeled_map[seg_id] = out
                if out["character"] not in ("UNKNOWN", "NARRATOR", ""):
                    known_characters.add(out["character"])

    # Fill any remaining segments as UNKNOWN for completeness.
    ordered: List[Dict[str, object]] = []
    for seg in segments:
        seg_id = str(seg.get("segment_id"))
        labeled = labeled_map.get(seg_id)
        if labeled is None:
            labeled = dict(seg)
            labeled.update(
                {
                    "role_type": "NARRATION",
                    "character": "UNKNOWN",
                }
            )
        ordered.append(labeled)

    if ordered:
        from .utils import write_jsonl

        write_jsonl(out_path, ordered)
    else:
        if not out_path.exists():
            out_path.touch()
    return out_path


def label_segments(
    segments_path: Path,
    out_path: Path,
    config: AppConfig,
    character_list_path: Optional[Path] = None,
) -> Path:
    if config.labeling.mode.lower() == "windowed":
        return _label_segments_windowed(segments_path, out_path, config, character_list_path)
    return _label_segments_per_segment(segments_path, out_path, config, character_list_path)
