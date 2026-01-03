import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

from .config import AppConfig
from .llm_client import call_llm_json
from .logging_utils import get_logger
from .utils import read_jsonl, write_json


logger = get_logger(__name__)


def _system_prompt() -> str:
    return (
        "You are a strict JSON generator. Output ONLY valid JSON: "
        "{\"characters\":[{\"canonical\":\"...\",\"aliases\":[\"...\"]}]}"
    )


def _user_prompt(text: str) -> str:
    return (
        "Extract character names from the following novel text. "
        "Merge variants of the same character (full name vs surname vs given name, "
        "nicknames, titles, honorifics). Only include named people or named roles. "
        "Exclude narrator and pronouns. "
        "If you are unsure, do not include the name. "
        "\nText:\n"
        f"{text}"
    )


def _normalize_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    filtered = []
    for ch in name:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            filtered.append(ch)
    return "".join(filtered)


def _merge_entries(
    entries: List[Dict[str, object]], canonical: str, aliases: List[str]
) -> None:
    def norm_set(values: List[str]) -> Set[str]:
        out: Set[str] = set()
        for v in values:
            n = _normalize_name(v)
            if n:
                out.add(n)
        return out

    cand_aliases = [canonical] + aliases
    cand_norm = norm_set(cand_aliases)
    if not cand_norm:
        return

    # Try to match existing entry by normalized overlap or substring.
    for entry in entries:
        existing = [str(entry.get("canonical", ""))] + list(entry.get("aliases", []))
        existing_norm = norm_set(existing)
        if cand_norm & existing_norm:
            entry_aliases = set(entry.get("aliases", []))
            entry_aliases.update(cand_aliases)
            entry["aliases"] = sorted(a for a in entry_aliases if a)
            return
        for cn in cand_norm:
            for en in existing_norm:
                if cn in en or en in cn:
                    entry_aliases = set(entry.get("aliases", []))
                    entry_aliases.update(cand_aliases)
                    entry["aliases"] = sorted(a for a in entry_aliases if a)
                    return

    entries.append({"canonical": canonical, "aliases": sorted(a for a in aliases if a)})


def _clean_entry(entry: Dict[str, object]) -> Tuple[str, List[str]]:
    canonical = str(entry.get("canonical", "")).strip()
    aliases = [str(a).strip() for a in entry.get("aliases", []) if str(a).strip()]
    if not canonical:
        if aliases:
            canonical = aliases[0]
            aliases = aliases[1:]
    return canonical, aliases


def discover_characters(
    segments_path: Path, out_path: Path, config: AppConfig
) -> Path:
    if out_path.exists():
        logger.info("Using existing character list: %s", out_path)
        return out_path

    segments = read_segments(segments_path)
    if not segments:
        write_json(out_path, {"characters": []})
        return out_path

    window_size = max(1, config.character_discovery.window_size)
    overlap = max(0, config.character_discovery.window_overlap)
    step = max(1, window_size - overlap)

    merged: List[Dict[str, object]] = []
    for start in tqdm(range(0, len(segments), step), desc="Character discovery", unit="window"):
        window = segments[start : start + window_size]
        if not window:
            continue
        text = "\n".join(seg.get("text", "") for seg in window)
        if not text.strip():
            continue

        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(text)},
        ]
        llm_out = call_llm_json(config, messages)
        items = llm_out.get("characters", [])
        if not isinstance(items, list):
            continue
        for item in items:
            canonical, aliases = _clean_entry(item)
            if not canonical:
                continue
            _merge_entries(merged, canonical, aliases)

    # Normalize and dedupe aliases.
    for entry in merged:
        aliases = [str(a).strip() for a in entry.get("aliases", []) if str(a).strip()]
        aliases = sorted(set(aliases))
        entry["aliases"] = aliases

    write_json(out_path, {"characters": merged})
    logger.info("Discovered %s characters", len(merged))
    return out_path


def read_segments(path: Path) -> List[Dict[str, str]]:
    segments: List[Dict[str, str]] = []
    for obj in read_jsonl(path):
        segments.append({"text": str(obj.get("text", ""))})
    return segments
