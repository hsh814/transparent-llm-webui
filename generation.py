"""Generation workers outlive browser connections; SQLite owns turn state."""

import json
import logging
import threading
import time
from datetime import datetime, timezone

import httpx

import db
import ollama
from tool_events import MAX_TOOL_CALLS, pending_calls, total_usage

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_workers: set[int] = set()


def error_message(exc: Exception) -> str:
    if isinstance(exc, ollama.ToolCallError):
        return str(exc)
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


def _reload(job: dict) -> dict | None:
    current = db.get_generation(job["session_id"], job["user_message_id"])
    if current is None or current["status"] != "running" or current["attempt"] != job["attempt"]:
        return None
    return current


def _call_arguments(call: dict) -> tuple[str, dict]:
    if (not isinstance(call, dict) or call.get("type") != "function"
            or not isinstance(call.get("id"), str) or not call["id"].strip()
            or not isinstance(call.get("function"), dict)):
        raise ollama.ToolCallError("The model returned an invalid tool call. Retry the response.")
    function = call["function"]
    name = function.get("name")
    if not isinstance(name, str) or not name.strip() or not isinstance(function.get("arguments"), str):
        raise ollama.ToolCallError("The model returned an incomplete tool call. Retry the response.")
    try:
        arguments = json.loads(function["arguments"])
    except (ValueError, TypeError) as exc:
        raise ollama.ToolCallError("The model returned invalid tool arguments. Retry the response.") from exc
    if not isinstance(arguments, dict):
        raise ollama.ToolCallError("The model returned invalid tool arguments. Retry the response.")
    return name, arguments


def _search_arguments(job: dict, call: dict) -> tuple[str, int]:
    name, arguments = _call_arguments(call)
    if any(event.get("origin") == "legacy" and any(
        previous["id"] == call["id"] for previous in event["message"].get("tool_calls", [])
    ) for event in job["trace"] if event["kind"] == "assistant"):
        raise ollama.ToolCallError(
            "This unfinished legacy app search cannot resume automatically. "
            "Regenerate the response or fork before its search call."
        )
    if name != "web_search":
        raise ollama.ToolCallError("The model requested an unsupported tool. That tool was not executed.")
    if not any(tool.get("type") == "function" and tool.get("function", {}).get("name") == name
               for tool in job.get("tools", [])):
        raise ollama.ToolCallError(
            "Web search is not allowed for this response. Enable Allow web search and retry, "
            "or fork before the tool call."
        )
    query = arguments.get("query")
    max_results = arguments.get("max_results", 5)
    if (set(arguments) - {"query", "max_results"} or not isinstance(query, str)
            or not query.strip() or isinstance(max_results, bool)
            or not isinstance(max_results, int) or not 1 <= max_results <= 5):
        raise ollama.ToolCallError(
            "The model requested an invalid web search. A nonempty query and 1–5 results are required."
        )
    return query, max_results


def _execute_tools(job: dict) -> dict | None:
    calls = pending_calls(job["trace"])
    retained_results = {
        event["message"]["tool_call_id"] for event in job["trace"]
        if event["kind"] == "tool" and event.get("retained") and event.get("message") is not None
        and event["status"] in ("completed", "failed")
    }
    call_count = sum(
        call["id"] not in retained_results
        for event in job["trace"] if event["kind"] == "assistant"
        for call in event["message"].get("tool_calls", [])
    )
    if call_count > MAX_TOOL_CALLS:
        raise ollama.ToolCallError(
            "The model exceeded the limit of eight tool calls for one response. "
            "Fork before the tool call or regenerate the response."
        )
    validated = []
    # Validate the whole pending batch before allowing any external operation.
    for call in calls:
        query, max_results = _search_arguments(job, call)
        validated.append((call, query, max_results))
    for call, query, max_results in validated:
        if not db.start_tool(job, call["id"]):
            return None
        job = _reload(job)
        if job is None:
            return None
        try:
            results = ollama.web_search(query, max_results=max_results)
        except Exception as exc:
            message = "Web search failed. " + error_message(exc)
            if not db.finish_tool(job, call["id"], {"error": message}, failed=True):
                return None
            job = _reload(job)
            if job is None:
                return None
            raise ollama.ToolCallError(message) from exc
        result = {"query": query, "max_results": max_results, "results": results,
                  "fetched_at": datetime.now(timezone.utc).isoformat()}
        if not db.finish_tool(job, call["id"], result):
            return None
        job = _reload(job)
        if job is None:
            return None
    return job


def _generate(job: dict) -> None:
    content = ""
    reasoning = ""
    try:
        while True:
            job = _execute_tools(job)
            if job is None:
                return
            # CAS immediately before each model call, including resumed tool rounds.
            if not db.update_generation(job, content, reasoning):
                return
            stream = ollama.chat_stream(
                job["model"], job["messages"], job["params"], tools=job.get("tools", [])
            )
            finished = False
            usage = None
            calls = []
            finish_reason = None
            last_flush = 0.0
            try:
                for chunk in stream:
                    if chunk["type"] == "content":
                        content += chunk["text"]
                    elif chunk["type"] == "reasoning":
                        reasoning += chunk["text"]
                    elif chunk["type"] == "done":
                        usage = chunk.get("usage")
                        calls = chunk.get("tool_calls") or []
                        finish_reason = chunk.get("finish_reason")
                        finished = True
                    now = time.monotonic()
                    if now - last_flush >= 0.1 or finished:
                        if not db.update_generation(job, content, reasoning):
                            return
                        last_flush = now
            finally:
                stream.close()
            if not finished:
                raise RuntimeError("Upstream closed without a complete response")
            if calls or finish_reason == "tool_calls":
                if not calls or finish_reason != "tool_calls":
                    raise ollama.ToolCallError("The model returned incomplete tool calls. Retry the response.")
                previous_ids = {
                    call["id"] for event in job["trace"] if event["kind"] == "assistant"
                    for call in event["message"].get("tool_calls", [])
                }
                for call in calls:
                    _call_arguments(call)
                    if call["id"] in previous_ids:
                        raise ollama.ToolCallError("The model reused a tool call ID. Retry the response.")
                    previous_ids.add(call["id"])
                message = {"role": "assistant", "content": content, "tool_calls": calls}
                if not db.record_tool_calls(job, message, reasoning, usage):
                    return
                content = reasoning = ""
                job = _reload(job)
                if job is None:
                    return
                continue
            if not (content or reasoning):
                raise RuntimeError("Upstream closed without a complete response")
            db.finish_generation(job, content, reasoning, total_usage(job["trace"], usage))
            return
    except Exception as exc:
        logger.warning("Generation failed for session %s (%s)", job["session_id"], type(exc).__name__)
        db.update_generation(job, content, reasoning)
        db.interrupt_generations(job["session_id"], error_message(exc), job=job)
