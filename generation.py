"""Generation workers outlive browser connections; SQLite owns turn state."""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx

import db
import ollama
from tool_events import MAX_TOOL_CALLS, budgeted_call_ids, pending_calls, total_usage

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


class _InvalidToolArguments(ollama.ToolCallError):
    """A complete call whose arguments the model can correct."""


def _tool_arguments(job: dict, call: dict) -> dict:
    name, arguments = _call_arguments(call)
    if any(event.get("origin") == "legacy" and any(
        previous["id"] == call["id"] for previous in event["message"].get("tool_calls", [])
    ) for event in job["trace"] if event["kind"] == "assistant"):
        raise ollama.ToolCallError(
            "This unfinished legacy app search cannot resume automatically. "
            "Regenerate the response or fork before its search call."
        )
    if name not in ("web_search", "web_fetch"):
        raise ollama.ToolCallError("The model requested an unsupported tool. That tool was not executed.")
    problems = []
    if name == "web_search":
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        if extra := set(arguments) - {"query", "max_results"}:
            problems.append("Unsupported arguments: " + ", ".join(sorted(extra)) + ". Only query and max_results are allowed.")
        if not isinstance(query, str) or not query.strip():
            problems.append("query must be a nonempty string.")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 5:
            problems.append(
                "max_results must be an integer from 1 to 5 (this app's maximum is 5; default 5 if omitted); "
                f"received {json.dumps(max_results, ensure_ascii=False)}."
            )
        validated = {"query": query, "max_results": max_results}
    else:
        url = arguments.get("url")
        valid_url = False
        if isinstance(url, str) and url and "\\" not in url and not any(
                char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url):
            try:
                parsed = urlsplit(url)
                valid_url = (
                    parsed.scheme in ("http", "https") and bool(parsed.hostname)
                    and parsed.username is None and parsed.password is None
                    and (parsed.port is None or 1 <= parsed.port <= 65535)
                )
            except ValueError:
                pass
        if not valid_url:
            problems.append("url must be an absolute HTTP(S) URL with a hostname and valid port, "
                            "without credentials, backslashes, whitespace, or control characters.")
        if set(arguments) - {"url"}:
            problems.append("Only the url argument is allowed for web_fetch.")
        validated = {"url": url}
    if problems:
        raise _InvalidToolArguments(
            f"{name} was not executed. " + " ".join(problems)
            + f" Correct the arguments and call {name} again with a new tool call ID."
        )
    return validated


def _execute_tools(job: dict) -> dict | None:
    calls = pending_calls(job["trace"])
    eligible = set(budgeted_call_ids(job["trace"])[:MAX_TOOL_CALLS])
    permitted = job.get("search_permission", bool(job.get("tools")))
    validated = []
    # Check the whole batch's permission/protocol before any external operation.
    # Argument rejections are paired tool replies, not executable searches.
    for call in calls:
        arguments = None
        rejection = None
        error_type = "invalid_arguments"
        try:
            arguments = _tool_arguments(job, call)
        except _InvalidToolArguments as exc:
            rejection = str(exc)
        if call["id"] not in eligible:
            rejection = (
                f"{call['function']['name']} was not executed: the shared budget of {MAX_TOOL_CALLS} "
                "web-tool calls for this response is exhausted. Do not retry this call. "
                "Answer using the evidence already gathered and state any uncertainty; "
                "no further web search or fetch calls are available."
            )
            error_type = "search_budget_exhausted"
        elif not permitted:
            raise ollama.ToolCallError(
                "Web access is not allowed for this response. Enable Allow web search & fetch and retry, "
                "or fork before the tool call."
            )
        validated.append((call, arguments, rejection, error_type))
    for call, arguments, rejection, error_type in validated:
        if rejection is not None:
            if not db.reject_tool(job, call["id"], rejection, error_type=error_type):
                return None
            job = _reload(job)
            if job is None:
                return None
            continue
        assert arguments is not None  # Preflight produced either validated arguments or a rejection.
        name = call["function"]["name"]
        if not db.start_tool(job, call["id"]):
            return None
        job = _reload(job)
        if job is None:
            return None
        status = "completed"
        try:
            if name == "web_search":
                result = {**arguments, "results": ollama.web_search(
                    arguments["query"], max_results=arguments["max_results"],
                )}
            else:
                result = {**arguments, **ollama.web_fetch(arguments["url"])}
        except Exception as exc:
            if name == "web_fetch" and isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                status = "unavailable"
                result = {
                    "url": arguments["url"], "error_type": "http_not_found",
                    "status_code": 404, "executed": True,
                    "error": (
                        "Ollama returned HTTP 404 (Not Found) while fetching this URL; "
                        "no page content was retrieved. Use the evidence already gathered or choose "
                        "another relevant URL. Do not repeat this unchanged URL. "
                        "If the available evidence is insufficient, state the uncertainty."
                    ),
                }
            else:
                message = ("Web fetch failed. " if name == "web_fetch" else "Web search failed. ") + error_message(exc)
                if not db.finish_tool(job, call["id"], {"error": message}, status="failed"):
                    return None
                job = _reload(job)
                if job is None:
                    return None
                raise ollama.ToolCallError(message) from exc
        result["attempted_at" if status == "unavailable" else "fetched_at"] = datetime.now(timezone.utc).isoformat()
        if not db.finish_tool(job, call["id"], result, status=status):
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
            # Save the exact offered schema, budget, and final-answer instruction.
            prepared = db.prepare_model_request(job)
            if prepared is None:
                return
            job = prepared
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
                final_answer_round = job.get("search_budget", {}).get("remaining") == 0
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
                if final_answer_round:
                    resolved = _execute_tools(job)
                    if resolved is None:
                        return
                    job = resolved
                    raise ollama.ToolCallError(
                        "The model requested tools after web access was disabled for the final answer. "
                        "No additional web requests were executed. Retry to request the final answer again."
                    )
                continue
            if not (content or reasoning):
                raise RuntimeError("Upstream closed without a complete response")
            db.finish_generation(job, content, reasoning, total_usage(job["trace"], usage))
            return
    except Exception as exc:
        logger.warning("Generation failed for session %s (%s)", job["session_id"], type(exc).__name__)
        db.update_generation(job, content, reasoning)
        db.interrupt_generations(job["session_id"], error_message(exc), job=job)
