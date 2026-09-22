"""Typed client wrapper over Ollama Cloud.

Two base URLs:
- native API:  https://ollama.com/api   (/api/tags, /api/show)
- OpenAI API:  https://ollama.com/v1    (/v1/models, /v1/chat/completions)

Auth: `Authorization: Bearer $OLLAMA_API_KEY`.
"""

import json
import os
import time
from functools import wraps
from typing import Iterator

import httpx

API_BASE = os.environ.get("OLLAMA_API_BASE_URL", "https://ollama.com")
OPENAI_BASE = os.environ.get("OLLAMA_OPENAI_BASE_URL", "https://ollama.com/v1")

TIMEOUT = httpx.Timeout(120.0, connect=20.0)
METADATA_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

_CACHE: dict[tuple, tuple[float, object]] = {}


def _metadata_cache(function):
    """Cache successful metadata for five minutes and outages for 30 seconds."""
    @wraps(function)
    def cached(*args):
        key = (function.__name__, args)
        entry = _CACHE.get(key)
        if entry and entry[0] > time.monotonic():
            if isinstance(entry[1], Exception):
                raise entry[1]
            return entry[1]
        try:
            value = function(*args)
        except Exception as exc:
            _CACHE[key] = (time.monotonic() + 30, exc)
            raise
        _CACHE[key] = (time.monotonic() + 300, value)
        return value
    return cached


def clear_cache() -> None:
    """Drop all cached values; next call re-fetches from Ollama."""
    _CACHE.clear()


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY', '')}"}


def _client() -> httpx.Client:
    return httpx.Client(base_url=OPENAI_BASE, headers=_headers(), timeout=TIMEOUT)


@_metadata_cache
def list_models() -> list[str]:
    """GET /v1/models -> [model ids]."""
    with _client() as client:
        resp = client.get("/models", timeout=METADATA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    return models


@_metadata_cache
def model_context_length(model: str) -> int | None:
    """POST /api/show -> context length from model_info, or None."""
    with httpx.Client(base_url=API_BASE, headers=_headers(), timeout=METADATA_TIMEOUT) as client:
        resp = client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        data = resp.json()
    info = data.get("model_info") or {}
    ctx = None
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, (int, float)):
            ctx = int(value)
            break
    return ctx


def _build_payload(model: str, messages: list[dict], params: dict) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    for key in ("reasoning_effort", "temperature", "top_p", "max_tokens", "seed"):
        if params.get(key) is not None:
            payload[key] = params[key]
    return payload


def chat_stream(model: str, messages: list[dict], params: dict) -> Iterator[dict]:
    """Stream chat completions.

    Yields {"type": "reasoning"|"content"|"done", "text": ...} per chunk;
    the final "done" chunk also carries "usage" (dict or None).
    """
    payload = _build_payload(model, messages, params)
    payload["stream_options"] = {"include_usage": True}
    usage: dict | None = None
    finished = False
    with _client() as client:
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Invalid response from Ollama") from exc
                if chunk.get("error"):
                    raise RuntimeError("Ollama reported a generation error")
                if chunk.get("usage"):
                    usage = chunk.get("usage")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                finish = choices[0].get("finish_reason")
                if finish is not None:
                    finished = True
                if delta.get("reasoning") or delta.get("reasoning_content"):
                    yield {"type": "reasoning", "text": delta.get("reasoning") or delta["reasoning_content"]}
                if delta.get("content"):
                    yield {"type": "content", "text": delta["content"]}
            if not finished:
                raise RuntimeError("Ollama disconnected before completing the response")
            yield {"type": "done", "text": "", "usage": usage}


def chat_once(model: str, messages: list[dict], params: dict) -> tuple[str, str | None, dict | None]:
    """Non-streaming chat; returns (content, reasoning_or_None, usage_or_None)."""
    payload = _build_payload(model, messages, params)
    payload["stream"] = False
    with _client() as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    return message.get("content", ""), message.get("reasoning"), data.get("usage")
