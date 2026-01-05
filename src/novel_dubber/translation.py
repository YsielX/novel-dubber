import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .config import AppConfig
from .llm_client import call_llm_json
from .logging_utils import get_logger
from .utils import append_jsonl, read_json, read_jsonl


logger = get_logger(__name__)


def _system_prompt() -> str:
    return (
        "You are a strict JSON generator. Output ONLY valid JSON: {\"translation\":\"...\"}. "
        "Do not add extra fields or commentary."
    )


def _system_prompt_window() -> str:
    return (
        "You are a strict JSON generator. Output ONLY valid JSON: "
        "{\"segments\":[{\"segment_id\":\"...\",\"translation\":\"...\"}]}. "
        "Do not add extra fields or commentary."
    )


def _user_prompt(text: str, role_type: str, target_lang: str, glossary: Optional[Dict[str, str]]) -> str:
    style = (
        "NARRATION: literary, neutral tone."
        if role_type == "NARRATION"
        else "DIALOGUE: keep personality, do not add content, keep names consistent."
    )
    glossary_note = f"\nGlossary: {glossary}" if glossary else ""
    return (
        f"Translate to {target_lang}. {style}"
        f"\nText: {text}"
        f"{glossary_note}"
    )


def _user_prompt_window(
    segments: List[Dict[str, str]],
    target_lang: str,
    glossary: Optional[Dict[str, str]],
) -> str:
    glossary_note = f"\nGlossary: {glossary}" if glossary else ""
    payload = json.dumps(segments, ensure_ascii=True)
    return (
        f"Translate to {target_lang}. "
        "Use role_type to guide style: "
        "NARRATION = literary neutral prose; "
        "DIALOGUE = natural speech, keep personality, no extra content. "
        "Preserve names and character labels. "
        f"\nSegments: {payload}"
        f"{glossary_note}"
    )


def translate_segments(
    segments_path: Path,
    out_path: Path,
    target_lang: str,
    config: AppConfig,
) -> Path:
    segments = read_jsonl(segments_path)
    existing = read_jsonl(out_path) if out_path.exists() else []
    done_ids = {seg.get("segment_id") for seg in existing}
    errors_path = out_path.with_name("translation_errors.jsonl")

    glossary = None
    if config.translation.glossary_path:
        glossary_path = Path(config.translation.glossary_path)
        if glossary_path.exists():
            glossary = read_json(glossary_path)

    if config.translation.mode.lower() != "windowed":
        for seg in tqdm(segments, desc="Translating", unit="segment"):
            seg_id = seg.get("segment_id")
            if seg_id in done_ids:
                continue
            role_type = str(seg.get("role_type", "NARRATION"))
            messages = [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": _user_prompt(
                        str(seg.get("text", "")),
                        role_type,
                        target_lang,
                        glossary,
                    ),
                },
            ]
            try:
                llm_out = call_llm_json(config, messages)
            except Exception as exc:
                logger.warning("Translation failed for %s: %s", seg_id, exc)
                append_jsonl(
                    errors_path,
                    [
                        {
                            "segment_id": seg_id,
                            "error": str(exc),
                        }
                    ],
                )
                continue
            translation = str(llm_out.get("translation", ""))
            out = dict(seg)
            out.update({"translation": translation, "target_lang": target_lang})
            append_jsonl(out_path, [out])
            done_ids.add(seg_id)

        if not out_path.exists():
            out_path.touch()
        return out_path

    window_size = max(1, config.translation.window_size)
    overlap = max(0, config.translation.window_overlap)
    step = max(1, window_size - overlap)

    for start in tqdm(range(0, len(segments), step), desc="Translating", unit="window"):
        window = segments[start : start + window_size]
        if not window:
            continue

        if not any(seg.get("segment_id") not in done_ids for seg in window):
            continue

        payload = [
            {
                "segment_id": str(seg.get("segment_id")),
                "role_type": str(seg.get("role_type", "NARRATION")),
                "character": str(seg.get("character", "UNKNOWN")),
                "text": str(seg.get("text", "")),
            }
            for seg in window
        ]
        messages = [
            {"role": "system", "content": _system_prompt_window()},
            {
                "role": "user",
                "content": _user_prompt_window(payload, target_lang, glossary),
            },
        ]
        try:
            llm_out = call_llm_json(config, messages)
        except Exception as exc:
            logger.warning("Translation window failed at %s: %s", start, exc)
            append_jsonl(
                errors_path,
                [
                    {
                        "window_start": start,
                        "segment_ids": [seg.get("segment_id") for seg in window],
                        "error": str(exc),
                    }
                ],
            )
            continue
        items = llm_out.get("segments", [])
        if not isinstance(items, list):
            continue

        window_results: List[Dict[str, object]] = []
        for item in items:
            seg_id = item.get("segment_id")
            if seg_id in done_ids or not seg_id:
                continue
            base = next((seg for seg in window if seg.get("segment_id") == seg_id), None)
            if not base:
                continue
            translation = str(item.get("translation", ""))
            out = dict(base)
            out.update({"translation": translation, "target_lang": target_lang})
            window_results.append(out)
            done_ids.add(seg_id)

        if window_results:
            append_jsonl(out_path, window_results)

    if not out_path.exists():
        out_path.touch()
    return out_path
