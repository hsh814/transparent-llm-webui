"""Generation workers outlive browser connections; SQLite owns turn state."""

import json
import logging
import threading
import time

import httpx

import db
import ollama

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_workers: set[int] = set()


def error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return "Ollama rejected the API key. Check OLLAMA_API_KEY and retry."
        if status == 429:
            return "Ollama's rate limit was reached. Wait a moment, then retry."
        return f"Ollama returned HTTP {status}. Check the selected model and settings, then retry."
    if isinstance(exc, httpx.TimeoutException):
        return "Ollama timed out. Your input is saved; retry when ready."
    if isinstance(exc, httpx.RequestError):
        return "Could not reach Ollama. Check your connection and retry."
    return "The response could not be completed. Your input is saved; retry when ready."


def start(session_id: int) -> None:
    with _lock:
        if session_id in _workers:
            return
        _workers.add(session_id)
        threading.Thread(target=_run, args=(session_id,), daemon=True).start()


def _run(session_id: int) -> None:
    try:
        while job := db.claim_generation(session_id):
            _generate(job)
    finally:
        with _lock:
            _workers.discard(session_id)
        # A retry may have arrived between the final claim and worker cleanup.
        if any(j["status"] == "queued" for j in db.list_generations(session_id)):
            start(session_id)


def _generate(job: dict) -> None:
    content = ""
    reasoning = ""
    last_flush = 0.0
    stream = None
    try:
        stream = ollama.chat_stream(job["model"], json.loads(job["messages_json"]), json.loads(job["params_json"]))
        finished = False
        usage = None
        for chunk in stream:
            if chunk["type"] == "content":
                content += chunk["text"]
            elif chunk["type"] == "reasoning":
                reasoning += chunk["text"]
            elif chunk["type"] == "done":
                usage = chunk.get("usage")
                finished = True
            now = time.monotonic()
            if now - last_flush >= 0.1 or finished:
                if not db.update_generation(job, content, reasoning):
                    return
                last_flush = now
        if not finished or not (content or reasoning):
            raise RuntimeError("Upstream closed without a complete response")
        db.finish_generation(job, content, reasoning, usage)
    except Exception as exc:
        logger.warning("Generation failed for session %s (%s)", job["session_id"], type(exc).__name__)
        db.update_generation(job, content, reasoning)
        db.interrupt_generations(job["session_id"], error_message(exc), job=job)
    finally:
        if stream is not None:
            stream.close()
