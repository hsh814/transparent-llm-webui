"""Chat routes: send (POST returns bubble), SSE stream (GET), delete, refresh.

Transport model (htmx 2.0 SSE extension): the extension only creates native
EventSource connections via `sse-connect`; it does not consume SSE returned
from an hx-post. So:

1. POST /sessions/{id}/send persists the user message and returns the HTML of
   the user bubble + an empty assistant bubble whose root carries
   `sse-connect="/sessions/{id}/stream?since=<user_msg_id>"`.
2. htmx swaps the bubbles in; the extension opens the EventSource; the stream
   endpoint runs the generation (reusing ollama.chat_stream), emits
   `reasoning`/`content` events that append into the bubble's zones, persists
   the assistant message, then emits `done` (which closes the source via
   `sse-close` and swaps in the finalized bubble).
"""

import json
import re

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

import db
import ollama
import templating

router = APIRouter()

_SENTENCE_END = r'(?<=[.!?;:。！？；：])\s*'


def _split_translation(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most `limit` chars.

    Priority: newlines → sentence punctuation → hard character break.
    Greedily accumulates pieces into chunks without exceeding the limit.
    """
    if limit <= 0 or len(text) <= limit:
        return [text] if text.strip() else []
    pieces: list[str] = []
    for line in text.split('\n'):
        if len(line) <= limit:
            pieces.append(line)
        else:
            for sub in re.split(_SENTENCE_END, line):
                if len(sub) <= limit:
                    pieces.append(sub)
                else:
                    for i in range(0, len(sub), limit):
                        pieces.append(sub[i:i + limit])
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
            current += "\n" + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _build_messages(folder: dict, session: dict) -> list[dict]:
    """Assemble the API message list from persisted rows.

    The folder's system prompt is mirrored into exactly one `system` row
    (visible in the UI and in the DB) and sent verbatim. No hidden prompt is
    ever added.
    """
    prompt = folder.get("system_prompt", "")
    if prompt:
        db.set_system_message(session["id"], prompt)
    else:
        db.delete_system_message(session["id"])
    return [
        {"role": row["role"], "content": row["content"]}
        for row in db.list_messages(session["id"])
        if row["role"] != "memo"
    ]


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _stream_headers() -> dict:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


@router.post("/sessions/{session_id}/send", response_class=HTMLResponse)
def send_message(request: Request, session_id: int, content: str = Form(...)):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    if folder is None:
        return HTMLResponse("", status_code=404)

    folder_type = folder.get("type", "chat")
    if folder_type == "memo":
        return HTMLResponse("", status_code=403)
    if folder_type == "translation":
        return _send_translation(request, session, folder, content)

    user_text = content.strip()
    if not user_text:
        return HTMLResponse("", status_code=400)

    user_row = db.add_message(
        session_id, "user", user_text, model=session["model"], reasoning_effort=None
    )
    params = json.loads(session.get("params_json") or "{}")

    user_bubble = templating.templates.env.get_template(
        "_message_bubble.html"
    ).render(
        request=request,
        message=user_row,
        session=session,
        streaming=False,
    )
    stream_bubble = templating.templates.env.get_template(
        "_stream_bubble.html"
    ).render(
        request=request,
        session=session,
        user_message_id=user_row["id"],
        effort=params.get("reasoning_effort"),
    )
    # Keep the on-screen system bubble in sync with the folder prompt.
    system_oob = ""
    for row in db.list_messages(session_id):
        if row["role"] == "system":
            bubble = templating.templates.env.get_template(
                "_message_bubble.html"
            ).render(request=request, message=row, session=session)
            system_oob = bubble.replace(
                'class="message system"',
                'class="message system" hx-swap-oob="true"',
                1,
            )
            break
    return system_oob + user_bubble + stream_bubble


def _send_translation(request: Request, session: dict, folder: dict,
                      content: str) -> HTMLResponse:
    """Translation send: split text into chunks, persist all user messages,
    return their bubbles plus a pending bubble that starts the translate chain.
    """
    text = content.strip()
    if not text:
        return HTMLResponse("", status_code=400)
    limit = folder.get("chunk_limit") or 1000
    chunks = _split_translation(text, limit)
    if not chunks:
        return HTMLResponse("", status_code=400)
    bubble_tpl = templating.templates.env.get_template("_message_bubble.html")
    parts: list[str] = []
    first_id: int | None = None
    for chunk in chunks:
        row = db.add_message(session["id"], "user", chunk,
                             model=session["model"], reasoning_effort=None)
        if first_id is None:
            first_id = row["id"]
        parts.append(bubble_tpl.render(request=request, message=row,
                                       session=session, streaming=False))
    pending = templating.templates.env.get_template("_translate_pending.html").render(
        request=request, session=session, user_message_id=first_id,
    )
    return HTMLResponse("".join(parts) + pending)


@router.post("/sessions/{session_id}/translate", response_class=HTMLResponse)
def translate_chunk(request: Request, session_id: int,
                    user_message_id: int = Form(...)):
    """Translate one chunk: [system_prompt, chunk] only — no history.

    Returns the finalized assistant bubble plus the next pending bubble
    (which auto-fires the next request), or the OOB usage badge on the last.
    """
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    if folder is None or folder.get("type") != "translation":
        return HTMLResponse("", status_code=403)
    user_msg = None
    for row in db.list_messages(session_id):
        if row["id"] == user_message_id and row["role"] == "user":
            user_msg = row
            break
    if user_msg is None:
        return HTMLResponse("", status_code=404)
    messages: list[dict] = []
    prompt = folder.get("system_prompt", "")
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": user_msg["content"]})
    params = json.loads(session.get("params_json") or "{}")
    try:
        content, reasoning, usage = ollama.chat_once(session["model"], messages, params)
    except Exception as exc:  # surface upstream failures to the UI
        return HTMLResponse(
            '<div class="message assistant"><div class="message-meta">'
            '<span class="role-label">assistant</span></div>'
            f'<div class="content">Translation error: {exc}</div></div>'
        )
    row = db.add_message(
        session_id, "assistant", content,
        reasoning=reasoning, model=session["model"],
        reasoning_effort=params.get("reasoning_effort"),
        prompt_tokens=usage.get("prompt_tokens") if usage else None,
        completion_tokens=usage.get("completion_tokens") if usage else None,
        total_tokens=usage.get("total_tokens") if usage else None,
    )
    bubble_tpl = templating.templates.env.get_template("_message_bubble.html")
    finalized = bubble_tpl.render(request=request, message=row, session=session,
                                  streaming=False)
    nxt = db.next_user_message(session_id, user_message_id)
    if nxt is not None:
        pending = templating.templates.env.get_template(
            "_translate_pending.html"
        ).render(request=request, session=session, user_message_id=nxt["id"])
        return HTMLResponse(finalized + pending)
    session_total = db.session_token_total(session_id)
    oob = templating.templates.env.get_template("_session_usage.html").render(
        request=request, session=session, total=session_total, oob=True)
    oob = " ".join(oob.split())
    return HTMLResponse(finalized + " " + oob)


@router.post("/sessions/{session_id}/memo", response_class=HTMLResponse)
def post_memo(request: Request, session_id: int, content: str = Form(...)):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    if folder is None or folder.get("type") != "memo":
        return HTMLResponse("", status_code=403)
    memo_text = content.strip()
    if not memo_text:
        return HTMLResponse("", status_code=400)
    row = db.add_message(session_id, "memo", memo_text)
    bubble = templating.templates.env.get_template(
        "_message_bubble.html"
    ).render(request=request, message=row, session=session, streaming=False)
    return HTMLResponse(bubble)


@router.get("/sessions/{session_id}/stream")
def stream_message(request: Request, session_id: int, since: int = 0):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    if folder is None:
        return HTMLResponse("", status_code=404)
    params = json.loads(session.get("params_json") or "{}")

    # Reconnect after completion: an assistant message already exists — replay it.
    for row in db.list_messages(session_id):
        if row["id"] > since and row["role"] == "assistant":
            def replay():
                if row["reasoning"]:
                    yield _sse("reasoning", row["reasoning"])
                yield _sse("content", row["content"])
                bubble = templating.templates.env.get_template(
                    "_message_bubble.html"
                ).render(request=request, message=row, session=session)
                bubble = " ".join(bubble.split())
                session_total = db.session_token_total(session_id)
                oob = templating.templates.env.get_template(
                    "_session_usage.html"
                ).render(request=request, session=session, total=session_total, oob=True)
                oob = " ".join(oob.split())
                yield _sse("done", bubble + " " + oob)

            return StreamingResponse(
                replay(), media_type="text/event-stream", headers=_stream_headers()
            )

    messages = _build_messages(folder, session)

    def generate():
        try:
            full_content: list[str] = []
            full_reasoning: list[str] = []
            usage: dict | None = None
            for chunk in ollama.chat_stream(session["model"], messages, params):
                ctype = chunk["type"]
                if ctype == "content":
                    full_content.append(chunk["text"])
                    yield _sse("content", chunk["text"])
                elif ctype == "reasoning":
                    full_reasoning.append(chunk["text"])
                    yield _sse("reasoning", chunk["text"])
                elif ctype == "done":
                    usage = chunk.get("usage")
                    break
            content = "".join(full_content)
            reasoning = "".join(full_reasoning) or None
            row = db.add_message(
                session_id,
                "assistant",
                content,
                reasoning=reasoning,
                model=session["model"],
                reasoning_effort=params.get("reasoning_effort"),
                prompt_tokens=usage.get("prompt_tokens") if usage else None,
                completion_tokens=usage.get("completion_tokens") if usage else None,
                total_tokens=usage.get("total_tokens") if usage else None,
            )
            bubble = templating.templates.env.get_template(
                "_message_bubble.html"
            ).render(request=request, message=row, session=session)
            # SSE: blank lines terminate an event — collapse the HTML to one line.
            bubble = " ".join(bubble.split())
            session_total = db.session_token_total(session_id)
            oob = templating.templates.env.get_template(
                "_session_usage.html"
            ).render(request=request, session=session, total=session_total, oob=True)
            oob = " ".join(oob.split())
            yield _sse("done", bubble + " " + oob)
        except Exception as exc:  # surface upstream failures to the UI
            error_bubble = (
                '<div class="message assistant"><div class="message-meta">'
                '<span class="role-label">assistant</span></div>'
                f'<div class="content">Error: {exc}</div></div>'
            )
            yield _sse("error", str(exc))
            yield _sse("done", error_bubble)

    return StreamingResponse(
        generate(), media_type="text/event-stream", headers=_stream_headers()
    )


@router.post(
    "/sessions/{session_id}/messages/{message_id}/delete", response_class=HTMLResponse
)
def delete_message(request: Request, session_id: int, message_id: int):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    db.delete_message(message_id)
    return templating.templates.TemplateResponse(
        request, "_chat_messages.html",
        {
            "folder": folder,
            "session": session,
            "messages": db.list_messages(session_id),
        },
    )


@router.get("/sessions/{session_id}/messages", response_class=HTMLResponse)
def messages_partial(request: Request, session_id: int):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    return templating.templates.TemplateResponse(
        request, "_chat_messages.html",
        {
            "folder": folder,
            "session": session,
            "messages": db.list_messages(session_id),
        },
    )
