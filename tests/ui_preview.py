"""Isolated browser QA server. Requires an explicit temporary CHAT_DB_PATH."""

import json
import os
import time

if not os.environ.get("CHAT_DB_PATH"):
    raise RuntimeError("Set CHAT_DB_PATH to a temporary database before running the preview.")

os.environ["OLLAMA_API_KEY"] = "browser-qa-placeholder"

import ollama

_failed_once: set[str] = set()
_response_versions: dict[str, int] = {}


def preview_stream(model, messages, params, *, tools=None):
    user_index = max(i for i, message in enumerate(messages) if message["role"] == "user")
    source = messages[user_index]["content"]
    if isinstance(source, list):
        source = "\n".join(part["text"] for part in source if part["type"] == "text")
    completed = sum(message["role"] == "tool" for message in messages[user_index + 1:])
    budget_search = "[budget-search]" in source
    correcting = "[invalid-search]" in source
    required = 2 if correcting or "[multi-search]" in source or "[two-round-search]" in source else 1
    if tools and "[no-search]" not in source and (budget_search or completed < required):
        indexes = range(completed + 1, completed + 4) if budget_search else range(1, 3) if "[multi-search]" in source else [completed + 1]
        calls = []
        for index in indexes:
            query = f"paper table figure evidence {index}"
            if "[search-fail]" in source:
                query += " [search-fail]"
            max_results = (10 if completed == 0 else 5) if correcting else 2
            calls.append({"id": f"search-{index}", "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": query, "max_results": max_results}),
            }})
        time.sleep(0.4)
        yield {"type": "reasoning", "text": "Choosing a focused search query."}
        yield {"type": "content", "text": "I will check the sources."}
        yield {"type": "done", "finish_reason": "tool_calls", "tool_calls": calls,
               "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}
        return
    if budget_search and not tools:
        time.sleep(0.4)
        saved_results = any(
            json.loads(message["content"]).get("results")
            for message in messages[user_index + 1:] if message["role"] == "tool"
        )
        answer = (
            "From the saved search evidence, Table 1 reports 42 samples and Figure 1 compares two methods. "
            "I cannot establish more detail from these snippets alone."
            if saved_results else "There is no saved search evidence, so I cannot establish an answer."
        )
        yield {"type": "content", "text": "No further tools are offered. " + answer}
        yield {"type": "done", "usage": {"prompt_tokens": 12, "completion_tokens": 28, "total_tokens": 40}}
        return
    if "[variants]" in source:
        version = _response_versions.get(source, 0) + 1
        _response_versions[source] = version
        time.sleep(0.4)
        yield {"type": "content", "text": f"Response version {version} for this input."}
        yield {"type": "done", "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
        return
    if "[fail]" in source and source not in _failed_once:
        _failed_once.add(source)
        time.sleep(0.3)
        raise RuntimeError("Simulated upstream failure")
    if "[instant]" in source:
        yield {"type": "content", "text": "An immediate response."}
    else:
        yield {"type": "reasoning", "text": "Checking the request.\nKeeping the original formatting."}
        lines = 80 if "[long]" in source else 40 if "[slow]" in source else 3
        text = "Here is a simulated response.\n\n" + "\n".join(
            f"{i + 1}. A useful point with enough detail to check reading and scrolling."
            for i in range(lines)
        )
        text += '\n\n    indented  code\n<script>window.unsafeExecuted = true</script>\nEnd of response.'
        for offset in range(0, len(text), 50):
            time.sleep(0.1 if "[slow]" in source else 0.025)
            yield {"type": "content", "text": text[offset:offset + 50]}
    yield {"type": "done", "usage": {"prompt_tokens": 12, "completion_tokens": 28, "total_tokens": 40}}


def preview_search(query, max_results=5):
    if type(max_results) is not int or not 1 <= max_results <= 5:
        raise AssertionError("An invalid search must not reach the provider.")
    time.sleep(0.4)
    if "[search-fail]" in query:
        raise RuntimeError("Simulated search failure")
    return [{"title": "Paper with a table and figure", "url": "https://example.org/paper",
             "content": "Table 1 reports 42 samples; Figure 1 compares the two methods."}]


ollama.list_models = lambda: ["gemma4:31b", "qwen3:32b", "gpt-oss:120b"]
ollama.model_context_length = lambda model: 32768
ollama.chat_stream = preview_stream
ollama.web_search = preview_search

from app import app  # noqa: E402 — the preview overrides are intentionally installed first.
