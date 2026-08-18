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

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

import db
import ollama
import templating

router = APIRouter()


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
                yield _sse("done", bubble)

            return StreamingResponse(
                replay(), media_type="text/event-stream", headers=_stream_headers()
            )

    messages = _build_messages(folder, session)

    def generate():
        try:
            full_content: list[str] = []
            full_reasoning: list[str] = []
            for chunk in ollama.chat_stream(session["model"], messages, params):
                ctype = chunk["type"]
                if ctype == "content":
                    full_content.append(chunk["text"])
                    yield _sse("content", chunk["text"])
                elif ctype == "reasoning":
                    full_reasoning.append(chunk["text"])
                    yield _sse("reasoning", chunk["text"])
                elif ctype == "done":
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
            )
            bubble = templating.templates.env.get_template(
                "_message_bubble.html"
            ).render(request=request, message=row, session=session)
            # SSE: blank lines terminate an event — collapse the HTML to one line.
            bubble = " ".join(bubble.split())
            yield _sse("done", bubble)
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
