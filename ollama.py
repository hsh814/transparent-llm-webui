"""Typed client wrapper over Ollama Cloud.

Two base URLs:
- native API:  https://ollama.com/api   (/api/tags, /api/show)
- OpenAI API:  https://ollama.com/v1    (/v1/models, /v1/chat/completions)

Auth: `Authorization: Bearer $OLLAMA_API_KEY`.
"""

import json
import os
import time
from typing import Iterator

import httpx

API_BASE = os.environ.get("OLLAMA_API_BASE_URL", "https://ollama.com")
OPENAI_BASE = os.environ.get("OLLAMA_OPENAI_BASE_URL", "https://ollama.com/v1")

TIMEOUT = httpx.Timeout(120.0, connect=20.0)

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300.0


def _cache_get(key: str) -> tuple[bool, object]:
    entry = _CACHE.get(key)
    if entry is None or time.monotonic() - entry[0] > _CACHE_TTL:
        return False, None
    return True, entry[1]


def _cache_set(key: str, value: object) -> None:
    _CACHE[key] = (time.monotonic(), value)


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY', '')}"}


def _client() -> httpx.Client:
    return httpx.Client(base_url=OPENAI_BASE, headers=_headers(), timeout=TIMEOUT)


def list_models() -> list[str]:
    """GET /v1/models -> [model ids]."""
    hit, val = _cache_get("models")
    if hit:
        return val
    with _client() as client:
        resp = client.get("/models")
        resp.raise_for_status()
        data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    _cache_set("models", models)
    return models


def model_context_length(model: str) -> int | None:
    """POST /api/show -> context length from model_info, or None."""
    hit, val = _cache_get(f"ctx:{model}")
    if hit:
        return val
    with httpx.Client(base_url=API_BASE, headers=_headers(), timeout=TIMEOUT) as client:
        resp = client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        data = resp.json()
    info = data.get("model_info") or {}
    ctx = None
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, (int, float)):
            ctx = int(value)
            break
    _cache_set(f"ctx:{model}", ctx)
    return ctx


def _build_payload(model: str, messages: list[dict], params: dict) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    for key in ("reasoning_effort", "temperature", "top_p", "max_tokens"):
        if params.get(key) is not None:
            payload[key] = params[key]
    options: dict = {}
    for key in ("num_ctx", "top_k", "repeat_penalty", "seed"):
        if params.get(key) is not None:
            options[key] = params[key]
    if options:
        payload["options"] = options
    return payload


def chat_stream(model: str, messages: list[dict], params: dict) -> Iterator[dict]:
    """Stream chat completions.

    Yields {"type": "reasoning"|"content"|"done", "text": ...} per chunk.
    """
    payload = _build_payload(model, messages, params)
    with _client() as client:
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    yield {"type": "done", "text": ""}
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                finish = choices[0].get("finish_reason")
                if finish is not None:
                    yield {"type": "done", "text": ""}
                    return
                if delta.get("reasoning"):
                    yield {"type": "reasoning", "text": delta["reasoning"]}
                if delta.get("content"):
                    yield {"type": "content", "text": delta["content"]}


def chat_once(model: str, messages: list[dict], params: dict) -> tuple[str, str | None]:
    """Non-streaming chat; returns (content, reasoning_or_None)."""
    payload = _build_payload(model, messages, params)
    payload["stream"] = False
    with _client() as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    return message.get("content", ""), message.get("reasoning")
