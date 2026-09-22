"""Isolated browser QA server. Requires an explicit temporary CHAT_DB_PATH."""

import os
import time

if not os.environ.get("CHAT_DB_PATH"):
    raise RuntimeError("Set CHAT_DB_PATH to a temporary database before running the preview.")

os.environ["OLLAMA_API_KEY"] = "browser-qa-placeholder"

import ollama

_failed_once: set[str] = set()


def preview_stream(model, messages, params):
    source = messages[-1]["content"]
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


ollama.list_models = lambda: ["gemma4:31b", "qwen3:32b", "gpt-oss:120b"]
ollama.model_context_length = lambda model: 32768
ollama.chat_stream = preview_stream

from app import app  # noqa: E402 — the preview overrides are intentionally installed first.
