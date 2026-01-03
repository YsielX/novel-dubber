import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import AppConfig, load_api_key
from .logging_utils import get_logger
from .utils import append_jsonl


logger = get_logger(__name__)
_USAGE_TOTALS: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _resolve_base_url(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    marker = "/chat/completions"
    if endpoint.endswith(marker):
        return endpoint[: -len(marker)]
    return endpoint


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text}")
    return json.loads(text[start : end + 1])


def chat_completion(config: AppConfig, messages: List[Dict[str, str]]) -> str:
    api_key = load_api_key(config)
    base_url = _resolve_base_url(config.llm.endpoint)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=config.llm.timeout_sec,
        max_retries=config.llm.max_retries,
    )
    params = {
        "model": config.llm.model,
        "messages": messages,
        "temperature": config.llm.temperature,
    }
    if config.llm.max_tokens > 0:
        params["max_tokens"] = config.llm.max_tokens
    resp = client.chat.completions.create(**params)
    usage = None
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens": getattr(resp.usage, "total_tokens", 0),
        }
        _record_usage(
            config,
            usage,
        )
    content = resp.choices[0].message.content
    _record_chat(config, messages, content, usage)
    return content


def call_llm_json(config: AppConfig, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    content = chat_completion(config, messages)
    return _extract_json(content)


def _record_usage(config: AppConfig, usage: Optional[Dict[str, Any]]) -> None:
    if not usage:
        return
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    total = int(usage.get("total_tokens", 0))
    _USAGE_TOTALS["prompt_tokens"] += prompt
    _USAGE_TOTALS["completion_tokens"] += completion
    _USAGE_TOTALS["total_tokens"] += total

    logger.info(
        "LLM usage: prompt=%s completion=%s total=%s (cumulative=%s)",
        prompt,
        completion,
        total,
        _USAGE_TOTALS["total_tokens"],
    )

    if not config.llm.usage_log:
        return
    path = Path(config.llm.usage_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        path,
        [
            {
                "ts": time.time(),
                "model": config.llm.model,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "cumulative_total": _USAGE_TOTALS["total_tokens"],
            }
        ],
    )


def _record_chat(
    config: AppConfig,
    messages: List[Dict[str, str]],
    content: str,
    usage: Optional[Dict[str, Any]],
) -> None:
    if not config.llm.chat_log:
        return
    path = Path(config.llm.chat_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        path,
        [
            {
                "ts": time.time(),
                "model": config.llm.model,
                "messages": messages,
                "response": content,
                "usage": usage or {},
            }
        ],
    )
