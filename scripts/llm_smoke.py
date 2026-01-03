import argparse
import json
import os
import time
from pathlib import Path

import yaml
from openai import OpenAI


def _resolve_base_url(endpoint: str) -> str:
    endpoint = (endpoint or "").strip()
    marker = "/chat/completions"
    if endpoint.endswith(marker):
        return endpoint[: -len(marker)]
    return endpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompt", default="Return JSON {\"ok\":true}.")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-log", default="")
    args = parser.parse_args()

    cfg = {}
    path = Path(args.config)
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm = cfg.get("llm", {})

    endpoint = str(llm.get("endpoint", ""))
    model = str(llm.get("model", ""))
    api_env = str(llm.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key in env var: {api_env}")

    base_url = _resolve_base_url(endpoint)
    timeout_sec = int(llm.get("timeout_sec", 120))
    max_retries = int(llm.get("max_retries", 2))
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_sec,
        max_retries=max_retries,
    )

    system_prompt = "You are a strict JSON generator. Output ONLY valid JSON."
    user_prompt = args.prompt
    if args.prompt_file:
        data = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        system_prompt = data.get("system_prompt", system_prompt)
        user_prompt = data.get("user_prompt", user_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if args.stream:
        start = time.time()
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=args.max_tokens,
            stream=True,
        )
        chunks = []
        char_count = 0
        last_log = time.time()
        stream_file = None
        if args.stream_log:
            stream_file = Path(args.stream_log).open("a", encoding="utf-8")
        try:
            for event in stream:
                choice = event.choices[0]
                delta = getattr(choice, "delta", None)
                text = ""
                if delta and getattr(delta, "content", None):
                    text = delta.content
                if text:
                    chunks.append(text)
                    char_count += len(text)
                    print(text, end="", flush=True)
                    if stream_file:
                        stream_file.write(text)
                        stream_file.flush()
                now = time.time()
                if now - last_log >= 2.0:
                    print(f"\n[stream] {char_count} chars...", flush=True)
                    last_log = now
            elapsed = time.time() - start
        finally:
            if stream_file:
                stream_file.write("\n")
                stream_file.close()

        print(f"\nlatency_sec={elapsed:.2f}")
    else:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        elapsed = time.time() - start

        print(f"latency_sec={elapsed:.2f}")
        print("content:")
        print(resp.choices[0].message.content)
        if getattr(resp, "usage", None):
            print("usage:", resp.usage)


if __name__ == "__main__":
    main()
