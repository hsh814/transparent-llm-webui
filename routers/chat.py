"""Chat input, resumable SSE display, retries, notes, and request inspection."""

import asyncio
import json
import re
from html import escape

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

import db
import generation
import inputs
import templating

router = APIRouter()


def _split_translation(text: str, limit: int) -> list[str]:
    """Prefer line/sentence boundaries, preserving the source verbatim."""
    if limit <= 0:
        raise ValueError("Chunk limit must be positive")
    chunks = []
    while len(text) > limit:
        window = text[:limit]
        boundary = window.rfind("\n") + 1
        if not boundary:
            matches = list(re.finditer(r"[。！？；：]|[.!?;:](?:\s|$)", window))
            boundary = matches[-1].end() if matches else 0
        if not boundary:
            boundary = window.rfind(" ") + 1
        boundary = boundary or limit
        chunks.append(text[:boundary])
        text = text[boundary:]
    if text:
        chunks.append(text)
    return chunks


def _session(session_id: int) -> dict:
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, "This chat no longer exists.")
    return session


def _sse(event: str, data: str) -> str:
    # SSE needs one data field per line. Never collapse content whitespace.
    return f"event: {event}\n" + "\n".join("data: " + line for line in data.split("\n")) + "\n\n"


def _bubble(request: Request, session: dict, message: dict) -> str:
    return templating.templates.env.get_template("_message_bubble.html").render(
        request=request, session=session, message=message,
        is_last=message["id"] == (db.last_message(session["id"]) or {}).get("id"),
    )


def _usage(request: Request, session: dict) -> str:
    return templating.templates.env.get_template("_session_usage.html").render(
        request=request, session=session, total=db.session_token_total(session["id"]), oob=True,
    ) + templating.session_sidebar(request, session)


def _messages(request: Request, session: dict) -> str:
    return templating.templates.env.get_template("_chat_messages.html").render(
        request=request, session=session, folder=db.get_folder(session["folder_id"]),
        messages=templating.chat_messages(session["id"]),
    )


@router.post("/sessions/{session_id}/send", response_class=HTMLResponse)
def send_message(
    request: Request, session_id: int, content: str = Form(""),
    files: list[UploadFile] | None = File(None),
    web_search: bool = Form(False),
):
    session = _session(session_id)
    folder = db.get_folder(session["folder_id"])
    if folder["type"] == "memo":
        raise HTTPException(403, "Use Save note in a memo folder.")
    if folder["type"] != "chat" and (files or web_search):
        raise HTTPException(400, "Attachments and web access are available in Chat folders.")
    text = content.strip()
    if not text and not files:
        raise HTTPException(400, "Write a message or attach an image or PDF first.")
    chunks = _split_translation(text, folder["chunk_limit"] or 1000) if folder["type"] == "translation" else [text]
    try:
        attachments = inputs.prepare_uploads(files or [])
        rows = db.enqueue_turns(session_id, chunks, attachments=attachments, web_search=web_search)
    except inputs.AttachmentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except db.GenerationBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    generation.start(session_id)
    session = _session(session_id)
    parts = []
    jobs = {job["user_message_id"]: job for job in db.list_generations(session_id)}
    live = next((j["user_message_id"] for j in jobs.values() if j["status"] in ("queued", "running")), None)
    for row in rows:
        parts.append(_bubble(request, session, row))
        parts.append(templating.generation_bubble(request, session, jobs[row["id"]], connect=live == row["id"]))
    sidebar = templating.folder_list_inner(request, session["folder_id"], session_id)
    return "".join(parts) + f'<div id="folder-list" hx-swap-oob="innerHTML">{sidebar}</div>'


@router.get("/sessions/{session_id}/stream")
def stream_message(request: Request, session_id: int, since: int):
    session = _session(session_id)
    if db.get_generation(session_id, since) is None:
        raise HTTPException(404, "This response does not belong to this chat.")
    generation.start(session_id)

    async def events():
        snapshots = {}
        trace_snapshot = None
        ticks = 0
        while not await request.is_disconnected():
            job = db.get_generation(session_id, since)
            if job is None:
                yield _sse("done", "<div></div>")
                return
            if job["trace_id"] != trace_snapshot:
                trace_snapshot = job["trace_id"]
                yield _sse("trace", templating.trace_timeline(
                    session, job, busy=job["status"] in ("queued", "running"),
                ))
            for key in ("content", "reasoning"):
                if job[key] != snapshots.get(key):
                    # Replace even with empty text: tool rounds reset partial output.
                    snapshots[key] = job[key]
                    yield _sse(key, "<span>" + escape(job[key] or "") + "</span>")
            if job["status"] not in ("queued", "running"):
                # Only one EventSource per chat. Activate the next queued bubble
                # and reconcile chunks that finished while this client was away.
                jobs = db.list_generations(session_id)
                live = next((j["user_message_id"] for j in jobs if j["status"] in ("queued", "running")), None)
                following = "".join(templating.generation_bubble(request, session, j, oob=True, connect=live == j["user_message_id"])
                                    for j in jobs if j["user_message_id"] > since and j["batch_id"] == job["batch_id"])
                yield _sse("done", templating.generation_bubble(request, session, job) + following + _usage(request, session))
                return
            ticks += 1
            if ticks % 60 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@router.post("/sessions/{session_id}/stop", response_class=HTMLResponse)
def stop_generation(request: Request, session_id: int):
    session = _session(session_id)
    db.interrupt_generations(session_id, "Stopped. Your input and partial response are saved.", "cancelled")
    return _messages(request, session) + _usage(request, session)


@router.post("/sessions/{session_id}/retry", response_class=HTMLResponse)
def retry_generation(request: Request, session_id: int, web_search: bool = Form(False)):
    session = _session(session_id)
    try:
        if not db.retry_generations(session_id, web_search=web_search):
            raise HTTPException(409, "Only the latest unfinished request can be retried.")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    generation.start(session_id)
    return _messages(request, session) + templating.session_sidebar(request, session)


@router.get("/sessions/{session_id}/edit", response_class=HTMLResponse)
def edit_conversation(request: Request, session_id: int, focus: int | None = None, focus_tool: str = ""):
    _session(session_id)
    try:
        snapshot = db.edit_snapshot(session_id)
    except (db.GenerationBusy, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return templating.templates.TemplateResponse(request, "_batch_editor.html", {
        "snapshot": snapshot, "session": snapshot["session"],
        "folder": snapshot["folder"], "focus_message": focus, "focus_tool": focus_tool,
        "editor_rows": sorted(
            [dict(message, editor_kind="message") for message in snapshot["messages"]]
            + [dict(group, editor_kind="tool") for group in snapshot["tool_groups"]],
            key=lambda row: row["position"],
        ),
    })


@router.post("/sessions/{session_id}/edit", response_class=HTMLResponse)
def save_edited_conversation(
    request: Request, session_id: int, draft: str = Form(...),
    intent: str = Form("save"), web_search: bool = Form(False),
):
    _session(session_id)
    if intent not in ("save", "send"):
        raise HTTPException(422, "Choose Save as fork or Save & send.")
    try:
        parsed = json.loads(draft)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "The editor draft could not be read. Your edits are still open.") from exc
    try:
        fork = db.save_edited_fork(
            session_id, parsed, send=intent == "send", web_search=intent == "send" and web_search,
        )
    except (db.GenerationBusy, db.EditConflict) as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if intent == "send":
        generation.start(fork["id"])
    folder = db.get_folder(fork["folder_id"])
    return templating.folder_list_inner(
        request, current_folder_id=folder["id"], current_session_id=fork["id"],
    ) + templating.chat_surface_fragment(
        request, folder, fork, oob=True, editor_web_search=intent == "send" and web_search,
    )


@router.get("/sessions/{session_id}/messages/{message_id}/source", response_class=HTMLResponse)
def show_message_source(request: Request, session_id: int, message_id: int):
    session = _session(session_id)
    message = next((row for row in db.list_messages(session_id) if row["id"] == message_id), None)
    if message is None or not message.get("provenance"):
        raise HTTPException(404, "No edit provenance is stored for this message.")
    return templating.templates.TemplateResponse(request, "_source_viewer.html", {
        "session": session, "message": message, "provenance": message["provenance"],
    })


@router.post("/sessions/{session_id}/messages/{message_id}/fork", response_class=HTMLResponse)
def fork_conversation(request: Request, session_id: int, message_id: int):
    _session(session_id)
    try:
        fork = db.fork_session(session_id, message_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    folder = db.get_folder(fork["folder_id"])
    return templating.folder_list_inner(
        request, current_folder_id=folder["id"], current_session_id=fork["id"],
    ) + templating.chat_surface_fragment(request, folder, fork, oob=True)


@router.post("/sessions/{session_id}/messages/{message_id}/events/{event_index}/fork", response_class=HTMLResponse)
def fork_event(
    request: Request, session_id: int, message_id: int, event_index: int,
    trace_id: str = Form(...),
):
    _session(session_id)
    try:
        fork = db.fork_session(session_id, message_id, event_index=event_index, trace_id=trace_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    folder = db.get_folder(fork["folder_id"])
    return templating.folder_list_inner(
        request, current_folder_id=folder["id"], current_session_id=fork["id"],
    ) + templating.chat_surface_fragment(request, folder, fork, oob=True)


@router.post("/sessions/{session_id}/messages/{message_id}/respond", response_class=HTMLResponse)
def request_response(request: Request, session_id: int, message_id: int, web_search: bool = Form(False)):
    session = _session(session_id)
    try:
        db.request_response(session_id, message_id, web_search=web_search)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    generation.start(session_id)
    return _messages(request, session) + _usage(request, session)


@router.post("/sessions/{session_id}/messages/{message_id}/regenerate", response_class=HTMLResponse)
def regenerate_response(request: Request, session_id: int, message_id: int, web_search: bool = Form(False)):
    session = _session(session_id)
    try:
        db.regenerate_response(session_id, message_id, web_search=web_search)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    generation.start(session_id)
    return _messages(request, session) + _usage(request, session)


@router.post("/sessions/{session_id}/messages/{user_message_id}/variants/{assistant_message_id}/select", response_class=HTMLResponse)
def select_response(request: Request, session_id: int, user_message_id: int, assistant_message_id: int):
    session = _session(session_id)
    try:
        db.select_response_variant(session_id, user_message_id, assistant_message_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _messages(request, session) + _usage(request, session)


@router.post("/sessions/{session_id}/translate", response_class=HTMLResponse)
def translate_chunk(request: Request, session_id: int, user_message_id: int = Form(...)):
    """Compatibility endpoint: display a saved chunk without duplicating it."""
    session = _session(session_id)
    job = db.get_generation(session_id, user_message_id)
    if job is None:
        raise HTTPException(404, "Translation request not found.")
    generation.start(session_id)
    return templating.generation_bubble(request, session, job)


@router.post("/sessions/{session_id}/memo", response_class=HTMLResponse)
def post_memo(request: Request, session_id: int, content: str = Form(...)):
    session = _session(session_id)
    if db.get_folder(session["folder_id"])["type"] != "memo":
        raise HTTPException(403, "Notes can only be saved in memo folders.")
    if not content.strip():
        raise HTTPException(400, "Write a note first.")
    row = db.add_message(session_id, "memo", content.strip())
    sidebar = templating.folder_list_inner(request, session["folder_id"], session_id)
    return _bubble(request, session, row) + f'<div id="folder-list" hx-swap-oob="innerHTML">{sidebar}</div>'


@router.post("/sessions/{session_id}/messages/{message_id}/delete", response_class=HTMLResponse)
def delete_message(request: Request, session_id: int, message_id: int):
    session = _session(session_id)
    try:
        if not db.delete_last_message(session_id, message_id):
            raise HTTPException(403, "Only the latest message can be deleted.")
    except db.GenerationBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    return _messages(request, session) + _usage(request, session)


@router.get("/sessions/{session_id}/messages", response_class=HTMLResponse)
def messages_partial(request: Request, session_id: int):
    return _messages(request, _session(session_id))


@router.get("/sessions/{session_id}/messages/{message_id}/prompt", response_class=HTMLResponse)
def show_prompt(request: Request, session_id: int, message_id: int):
    session = _session(session_id)
    folder = db.get_folder(session["folder_id"])
    rows = db.list_messages(session_id)
    message = next((m for m in rows if m["id"] == message_id and m["role"] in ("user", "assistant")), None)
    if message is None:
        raise HTTPException(404, "Request or response not found.")
    job = db.get_message_generation(session_id, message_id)
    provenance = message.get("provenance")
    historical = provenance if message["role"] == "assistant" else None
    if historical:
        original = historical.get("source_request")
        job = {
            "messages": original.get("messages", []), "params": original.get("params", {}),
            "tools": original.get("tools", []), "request": original,
            "model": original.get("model") or historical.get("source_model"),
            "status": "historical", "user_message_id": None,
        } if original else None
    if job is None and message["role"] == "user":
        raise HTTPException(404, "The original request was not stored for this message.")
    note = (
        "Actual original response request, not the edited fork context. The saved response text or Thinking is "
        + ("manually edited, not a new model output." if historical["kind"] == "manual" else "carried from the earlier context.")
        if historical and job else
        "The actual original response request was not recorded; no replacement request has been invented."
        if historical else
        "Current saved request for the edited input. This is not a claim that it has been sent to the model."
        if provenance and message["role"] == "user" else None
    )
    if job is not None:
        messages = job["messages"]
    elif historical:
        messages = []
    else:
        note = "Legacy response: this is a best-effort reconstruction; the original request was not stored."
        visible = db.visible_messages(session_id)
        messages = [{"role": m["role"], "content": m["content"]} for m in visible
                    if m["id"] < message_id and m["role"] in ("user", "assistant")]
        if folder["type"] == "translation":
            users = [m for m in rows if m["role"] == "user"]
            index = [m for m in rows if m["role"] == "assistant"].index(message)
            messages = [{"role": "user", "content": users[index]["content"]}] if index < len(users) else []
        prompt = db.get_system_prompt_by_hash(message["system_prompt_hash"]) if message["system_prompt_hash"] else folder["system_prompt"]
        if prompt:
            messages.insert(0, {"role": "system", "content": prompt})
    user = next((m for m in rows if job and m["id"] == job["user_message_id"]), None)
    return templating.templates.TemplateResponse(request, "_prompt_viewer.html", {
        "session": session, "message": message, "messages": messages,
        "messages_json": json.dumps(messages, ensure_ascii=False, indent=2),
        "sys_note": note, "recorded": job is not None,
        "request_params": job["params"] if job else None,
        "job": job,
        "attachments": user["attachments"] if user else [],
        "request_tools": job.get("tools", []) if job else [],
        "request_snapshot": job.get("request") if job else None,
    })


@router.get("/sessions/{session_id}/messages/{message_id}/events/{event_index}/prompt", response_class=HTMLResponse)
def show_event_prompt(
    request: Request, session_id: int, message_id: int, event_index: int, trace_id: str,
):
    session = _session(session_id)
    try:
        job = db.get_event_generation(session_id, message_id, event_index, trace_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if job is None:
        raise HTTPException(404, "This tool-call request is no longer available at that trace boundary.")
    return templating.templates.TemplateResponse(request, "_prompt_viewer.html", {
        "session": session, "message": None, "messages": job["messages"],
        "messages_json": json.dumps(job["messages"], ensure_ascii=False, indent=2),
        "sys_note": "Exact saved model request for this assistant tool-call round. Thinking produced by this round is its output, not part of this round's input. Editing that output preserves this historical request; retained Thinking can be included in later requests.",
        "recorded": True, "request_params": job["params"], "job": job,
        "attachments": [], "request_tools": job.get("tools", []),
        "request_snapshot": job.get("request"),
    })
