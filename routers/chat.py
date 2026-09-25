"""Chat input, resumable SSE display, retries, notes, and request inspection."""

import asyncio
import json
import re
from html import escape

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

import db
import generation
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
def send_message(request: Request, session_id: int, content: str = Form(...)):
    session = _session(session_id)
    folder = db.get_folder(session["folder_id"])
    if folder["type"] == "memo":
        raise HTTPException(403, "Use Save note in a memo folder.")
    text = content.strip()
    if not text:
        raise HTTPException(400, "Write a message first.")
    chunks = _split_translation(text, folder["chunk_limit"] or 1000) if folder["type"] == "translation" else [text]
    try:
        rows = db.enqueue_turns(session_id, chunks)
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
        content_pos = reasoning_pos = ticks = 0
        while not await request.is_disconnected():
            job = db.get_generation(session_id, since)
            if job is None:
                yield _sse("done", "<div></div>")
                return
            for key, position in (("content", content_pos), ("reasoning", reasoning_pos)):
                if len(job[key]) > position:
                    # htmx inserts SSE as HTML: wrap escaped text to preserve whitespace.
                    yield _sse(key, "<span>" + escape(job[key]) + "</span>")
            content_pos, reasoning_pos = len(job["content"]), len(job["reasoning"])
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
def retry_generation(request: Request, session_id: int):
    session = _session(session_id)
    try:
        if not db.retry_generations(session_id):
            raise HTTPException(409, "Only the latest unfinished request can be retried.")
    except db.GenerationBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    generation.start(session_id)
    return _messages(request, session) + templating.session_sidebar(request, session)


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
    message = next((m for m in rows if m["id"] == message_id and m["role"] == "assistant"), None)
    if message is None:
        raise HTTPException(404, "Response not found.")
    job = next((j for j in db.list_generations(session_id) if j["assistant_message_id"] == message_id), None)
    note = None
    if job is not None:
        messages = json.loads(job["messages_json"])
    else:
        note = "Legacy response: this is a best-effort reconstruction; the original request was not stored."
        messages = [{"role": m["role"], "content": m["content"]} for m in rows
                    if m["id"] < message_id and m["role"] in ("user", "assistant")]
        if folder["type"] == "translation":
            users = [m for m in rows if m["role"] == "user"]
            index = [m for m in rows if m["role"] == "assistant"].index(message)
            messages = [{"role": "user", "content": users[index]["content"]}] if index < len(users) else []
        prompt = db.get_system_prompt_by_hash(message["system_prompt_hash"]) if message["system_prompt_hash"] else folder["system_prompt"]
        if prompt:
            messages.insert(0, {"role": "system", "content": prompt})
    return templating.templates.TemplateResponse(request, "_prompt_viewer.html", {
        "session": session, "message": message, "messages": messages,
        "messages_json": json.dumps(messages, ensure_ascii=False, indent=2),
        "sys_note": note, "recorded": job is not None,
        "request_params": json.loads(job["params_json"]) if job else None,
    })
