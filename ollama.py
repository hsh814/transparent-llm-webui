"""Provider-routed OpenAI-compatible chat and Ollama-native metadata/tools."""

import json
import os
import time
from functools import wraps
from hashlib import sha256
from inspect import signature
from typing import Iterator
from urllib.parse import urlsplit

import httpx

import providers

TIMEOUT = httpx.Timeout(120.0, connect=20.0)
METADATA_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
SEARCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_CACHE: dict[tuple, tuple[float, object]] = {}


def _metadata_cache(function):
    """Cache successful metadata for five minutes and outages for 30 seconds."""
    parameters = signature(function)
    @wraps(function)
    def cached(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        provider = bound.arguments["provider"]
        configured = providers.get_provider(provider)
        credential = sha256(os.environ[configured.api_key_env].encode()).digest()
        arguments = tuple(value for name, value in bound.arguments.items() if name != "provider")
        key = (configured, credential, function.__name__, arguments)
        entry = _CACHE.get(key)
        if entry and entry[0] > time.monotonic():
            if isinstance(entry[1], Exception):
                raise entry[1]
            return entry[1]
        try:
            value = function(*bound.args, **bound.kwargs)
        except Exception as exc:
            _CACHE[key] = (time.monotonic() + 30, exc)
            raise
        _CACHE[key] = (time.monotonic() + 300, value)
        return value
    return cached


def clear_cache(provider: str | None = None) -> None:
    """Drop metadata for one provider, or all providers when omitted."""
    if provider is None:
        _CACHE.clear()
    else:
        for key in list(_CACHE):
            if key[0].id == provider:
                _CACHE.pop(key, None)


def _headers(provider: str = "ollama") -> dict:
    configured = providers.get_provider(provider)
    return {"Authorization": f"Bearer {os.environ[configured.api_key_env]}"}


def _client(provider: str = "ollama") -> httpx.Client:
    configured = providers.get_provider(provider)
    return httpx.Client(base_url=configured.base_url, headers=_headers(provider), timeout=TIMEOUT)


def _native_client(provider: str, timeout: httpx.Timeout) -> httpx.Client:
    configured = providers.get_provider(provider)
    if configured.kind != "ollama":
        raise ToolCallError("Native web tools are not supported by the selected provider.")
    return httpx.Client(base_url=configured.native_base_url, headers=_headers(provider), timeout=timeout)


@_metadata_cache
def list_models(provider: str = "ollama") -> list[str]:
    """GET the selected provider's model catalogue."""
    with _client(provider) as client:
        resp = client.get("/models", timeout=METADATA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    return models


@_metadata_cache
def model_context_length(model: str, provider: str = "ollama") -> int | None:
    """Read native Ollama context metadata; other providers have no native metadata."""
    if providers.get_provider(provider).kind != "ollama":
        return None
    with _native_client(provider, METADATA_TIMEOUT) as client:
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


def web_search(query: str, max_results: int = 5, *, provider: str = "ollama") -> list[dict]:
    """Fetch validated source snippets once; malformed results fail visibly."""
    with _native_client(provider, SEARCH_TIMEOUT) as client:
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


def web_fetch(url: str, *, provider: str = "ollama") -> dict:
    """Retrieve provider-extracted page text; never contact the target URL locally."""
    with _native_client(provider, SEARCH_TIMEOUT) as client:
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
    model: str, messages: list[dict], params: dict, *, tools: list[dict] | None = None,
    provider: str = "ollama",
) -> Iterator[dict]:
    """Stream text/reasoning and return complete, validated calls on the done chunk."""
    payload = _build_payload(model, messages, params, tools=tools)
    payload["stream_options"] = {"include_usage": True}
    usage: dict | None = None
    finished = False
    finish_reason = None
    calls: dict[int, dict] = {}
    with _client(provider) as client:
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
                    raise RuntimeError("Invalid response from provider") from exc
                if not isinstance(chunk, dict):
                    raise RuntimeError("Invalid response from provider")
                if chunk.get("error"):
                    raise RuntimeError("Provider reported a generation error")
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
                        raise ToolCallError("Provider returned malformed tool calls. Retry the response.")
                    for fragment in fragments:
                        if not isinstance(fragment, dict):
                            raise ToolCallError("Provider returned malformed tool calls. Retry the response.")
                        index = fragment.get("index")
                        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                            raise ToolCallError("Provider returned a tool call without a valid index. Retry the response.")
                        call = calls.setdefault(index, {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                        })
                        if fragment.get("type") not in (None, "function"):
                            raise ToolCallError("Provider returned an unsupported tool call type.")
                        call_id = fragment.get("id")
                        if call_id is not None:
                            if not isinstance(call_id, str):
                                raise ToolCallError("Provider returned malformed tool calls. Retry the response.")
                            call["id"] += call_id
                        function = fragment.get("function") or {}
                        if not isinstance(function, dict):
                            raise ToolCallError("Provider returned malformed tool calls. Retry the response.")
                        for key in ("name", "arguments"):
                            value = function.get(key)
                            if value is not None:
                                if not isinstance(value, str):
                                    raise ToolCallError("Provider returned malformed tool calls. Retry the response.")
                                call["function"][key] += value
                if finish is not None:
                    finish_reason = finish
                    finished = True
                if delta.get("reasoning") or delta.get("reasoning_content"):
                    yield {"type": "reasoning", "text": delta.get("reasoning") or delta["reasoning_content"]}
                if delta.get("content"):
                    yield {"type": "content", "text": delta["content"]}
            if not finished:
                raise RuntimeError("Provider disconnected before completing the response")
            tool_calls = [calls[index] for index in sorted(calls)]
            if tool_calls or finish_reason == "tool_calls":
                if (not tool_calls or finish_reason != "tool_calls"
                        or sorted(calls) != list(range(len(calls)))):
                    raise ToolCallError("Provider returned incomplete tool calls. Retry the response.")
                ids = set()
                for call in tool_calls:
                    if not call["id"].strip() or not call["function"]["name"].strip() or call["id"] in ids:
                        raise ToolCallError("Provider returned incomplete or duplicate tool call IDs. Retry the response.")
                    ids.add(call["id"])
                    try:
                        arguments = json.loads(call["function"]["arguments"])
                    except (ValueError, TypeError) as exc:
                        raise ToolCallError("Provider returned invalid tool arguments. Retry the response.") from exc
                    if not isinstance(arguments, dict):
                        raise ToolCallError("Provider returned invalid tool arguments. Retry the response.")
            yield {"type": "done", "text": "", "usage": usage,
                   "tool_calls": tool_calls, "finish_reason": finish_reason}


def chat_once(
    model: str, messages: list[dict], params: dict, *, provider: str = "ollama"
) -> tuple[str, str | None, dict | None]:
    """Non-streaming chat; returns (content, reasoning_or_None, usage_or_None)."""
    payload = _build_payload(model, messages, params)
    payload["stream"] = False
    with _client(provider) as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    return message.get("content", ""), message.get("reasoning"), data.get("usage")
