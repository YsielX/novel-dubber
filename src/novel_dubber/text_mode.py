import re
from pathlib import Path
from typing import Dict, List

from .config import AppConfig
from .logging_utils import get_logger
from .utils import write_jsonl


logger = get_logger(__name__)


QUOTE_PAIRS = [
    ("\"", "\""),
    ("\u201c", "\u201d"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
]
SENTENCE_END = set("。！？.!?…；;！？")


def _strip_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_text_segments(text: str) -> List[Dict[str, str]]:
    segments: List[Dict[str, str]] = []
    buf: List[str] = []
    mode = "NARRATION"
    stack: List[str] = []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    open_quotes = {q[0]: q[1] for q in QUOTE_PAIRS}
    close_quotes = {q[1] for q in QUOTE_PAIRS}

    def flush(current_mode: str) -> None:
        if not buf:
            return
        chunk = _strip_ws("".join(buf))
        buf.clear()
        if not chunk:
            return
        segments.append({"role_type": current_mode, "text": chunk})

    i = 0
    while i < len(text):
        ch = text[i]
        if ch in open_quotes and (not stack or open_quotes.get(ch) != stack[-1]):
            flush(mode)
            stack.append(open_quotes[ch])
            mode = "DIALOGUE"
            i += 1
            continue
        if ch in close_quotes and stack and ch == stack[-1]:
            flush(mode)
            stack.pop()
            mode = "NARRATION" if not stack else "DIALOGUE"
            i += 1
            continue
        if ch in ("\n", "\r"):
            flush(mode)
            i += 1
            continue

        buf.append(ch)
        if ch in SENTENCE_END:
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            if next_ch and next_ch in SENTENCE_END:
                i += 1
                continue
            flush(mode)
        i += 1

    flush(mode)
    return segments


def add_pause_rules(segments: List[Dict[str, object]], config: AppConfig) -> None:
    for seg in segments:
        text = str(seg.get("text", ""))
        pause = config.text_dub.default_pause_sec
        if text:
            last = text[-1]
            if last in config.text_dub.punctuation_pause_sec:
                pause = float(config.text_dub.punctuation_pause_sec[last])
        seg["pause_after_sec"] = pause


def build_text_segments(
    text_path: Path,
    out_path: Path,
    config: AppConfig,
) -> Path:
    if out_path.exists():
        logger.info("Using existing text segments: %s", out_path)
        return out_path

    text = text_path.read_text(encoding="utf-8")
    segments = parse_text_segments(text)
    results: List[Dict[str, object]] = []
    for idx, seg in enumerate(segments):
        results.append(
            {
                "segment_id": f"seg_{idx:06d}",
                "text_index": idx,
                "role_type": seg["role_type"],
                "text": seg["text"],
            }
        )
    add_pause_rules(results, config)
    write_jsonl(out_path, results)
    return out_path
