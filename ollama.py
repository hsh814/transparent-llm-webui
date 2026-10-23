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
from urllib.parse import urlsplit

import httpx

API_BASE = os.environ.get("OLLAMA_API_BASE_URL", "https://ollama.com")
OPENAI_BASE = os.environ.get("OLLAMA_OPENAI_BASE_URL", "https://ollama.com/v1")

TIMEOUT = httpx.Timeout(120.0, connect=20.0)
METADATA_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
SEARCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

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


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Fetch validated source snippets once; malformed results fail visibly."""
    with httpx.Client(base_url=API_BASE, headers=_headers(), timeout=SEARCH_TIMEOUT) as client:
        resp = client.post("/api/web_search", json={"query": query, "max_results": max_results})
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict) or data.get("error") or not isinstance(data.get("results"), list):
        raise ValueError("Invalid search response from Ollama")
    if len(data["results"]) > max_results:
        raise ValueError("Ollama returned more search results than requested")
    results = []
    for result in data["results"]:
        if not isinstance(result, dict) or any(
            not isinstance(result.get(key), str) for key in ("title", "url", "content")
        ):
            raise ValueError("Invalid search result from Ollama")
        url = result["url"]
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname
                or any(character.isspace() or ord(character) < 32 for character in url)):
            raise ValueError("Invalid search source URL from Ollama")
        results.append({key: result[key] for key in ("title", "url", "content")})
    return results


def web_fetch(url: str) -> dict:
    """Retrieve provider-extracted page text; never contact the target URL locally."""
    with httpx.Client(base_url=API_BASE, headers=_headers(), timeout=SEARCH_TIMEOUT) as client:
        resp = client.post("/api/web_fetch", json={"url": url})
        resp.raise_for_status()
        data = resp.json()
    if (not isinstance(data, dict) or data.get("error")
            or not isinstance(data.get("title"), str) or not isinstance(data.get("content"), str)
            or not isinstance(data.get("links"), list)
            or any(not isinstance(link, str) for link in data["links"])):
        raise ValueError("Invalid web fetch response from Ollama")
    return {key: data[key] for key in ("title", "content", "links")}


class ToolCallError(ValueError):
    """A safe, user-visible rejection of an incomplete or invalid tool request."""


def _build_payload(
    model: str, messages: list[dict], params: dict, *, tools: list[dict] | None = None
) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    for key in ("reasoning_effort", "temperature", "top_p", "max_tokens", "seed"):
        if params.get(key) is not None:
            payload[key] = params[key]
    if tools:
        payload["tools"] = tools
    return payload


def chat_stream(
    model: str, messages: list[dict], params: dict, *, tools: list[dict] | None = None
) -> Iterator[dict]:
    """Stream text/reasoning and return complete, validated calls on the done chunk."""
    payload = _build_payload(model, messages, params, tools=tools)
    payload["stream_options"] = {"include_usage": True}
    usage: dict | None = None
    finished = False
    finish_reason = None
    calls: dict[int, dict] = {}
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
                if not isinstance(chunk, dict):
                    raise RuntimeError("Invalid response from Ollama")
                if chunk.get("error"):
                    raise RuntimeError("Ollama reported a generation error")
                if chunk.get("usage"):
                    usage = chunk.get("usage")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason")
                fragments = delta.get("tool_calls")
                if fragments is not None:
                    if not isinstance(fragments, list) or finish_reason is not None:
                        raise ToolCallError("Ollama returned malformed tool calls. Retry the response.")
                    for fragment in fragments:
                        if not isinstance(fragment, dict):
                            raise ToolCallError("Ollama returned malformed tool calls. Retry the response.")
                        index = fragment.get("index")
                        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                            raise ToolCallError("Ollama returned a tool call without a valid index. Retry the response.")
                        call = calls.setdefault(index, {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                        })
                        if fragment.get("type") not in (None, "function"):
                            raise ToolCallError("Ollama returned an unsupported tool call type.")
                        call_id = fragment.get("id")
                        if call_id is not None:
                            if not isinstance(call_id, str):
                                raise ToolCallError("Ollama returned malformed tool calls. Retry the response.")
                            call["id"] += call_id
                        function = fragment.get("function") or {}
                        if not isinstance(function, dict):
                            raise ToolCallError("Ollama returned malformed tool calls. Retry the response.")
                        for key in ("name", "arguments"):
                            value = function.get(key)
                            if value is not None:
                                if not isinstance(value, str):
                                    raise ToolCallError("Ollama returned malformed tool calls. Retry the response.")
                                call["function"][key] += value
                if finish is not None:
                    finish_reason = finish
                    finished = True
                if delta.get("reasoning") or delta.get("reasoning_content"):
                    yield {"type": "reasoning", "text": delta.get("reasoning") or delta["reasoning_content"]}
                if delta.get("content"):
                    yield {"type": "content", "text": delta["content"]}
            if not finished:
                raise RuntimeError("Ollama disconnected before completing the response")
            tool_calls = [calls[index] for index in sorted(calls)]
            if tool_calls or finish_reason == "tool_calls":
                if (not tool_calls or finish_reason != "tool_calls"
                        or sorted(calls) != list(range(len(calls)))):
                    raise ToolCallError("Ollama returned incomplete tool calls. Retry the response.")
                ids = set()
                for call in tool_calls:
                    if not call["id"].strip() or not call["function"]["name"].strip() or call["id"] in ids:
                        raise ToolCallError("Ollama returned incomplete or duplicate tool call IDs. Retry the response.")
                    ids.add(call["id"])
                    try:
                        arguments = json.loads(call["function"]["arguments"])
                    except (ValueError, TypeError) as exc:
                        raise ToolCallError("Ollama returned invalid tool arguments. Retry the response.") from exc
                    if not isinstance(arguments, dict):
                        raise ToolCallError("Ollama returned invalid tool arguments. Retry the response.")
            yield {"type": "done", "text": "", "usage": usage,
                   "tool_calls": tool_calls, "finish_reason": finish_reason}


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
