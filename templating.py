"""Jinja2 templates + fragment helpers shared by app and routers."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.templating import Jinja2Templates

import db
import ollama

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def asset_url(name: str) -> str:
    """Keep cached browser assets in sync with the server's current templates."""
    asset = Path(__file__).parent / "static" / name
    return f"/static/{name}?v={asset.stat().st_mtime_ns}"


templates.env.globals["asset_url"] = asset_url


def _from_json(value: str) -> dict:
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


templates.env.filters["from_json"] = _from_json


def _utc_iso(value: str) -> str:
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.isoformat()


templates.env.filters["utc_iso"] = _utc_iso


def _sort_models(models: list[str], usage: dict[str, int]) -> list[str]:
    """Used models first (desc session count), then unused by name."""
    return sorted(models, key=lambda m: (-usage.get(m, 0), m))


templates.env.filters["sort_models"] = _sort_models


def chat_context(session: dict) -> dict:
    """Live model list + context cap for a session, with safe fallbacks."""
    try:
        models = ollama.list_models()
    except Exception:
        models = [session["model"]]
    try:
        ctx_cap = ollama.model_context_length(session["model"])
    except Exception:
        ctx_cap = None
    return {"models": list(dict.fromkeys([session["model"], *models])), "ctx_cap": ctx_cap, "usage": db.model_usage()}


def chat_messages(session_id: int) -> list[dict]:
    rows, generations = db.conversation_snapshot(session_id)
    jobs = {job["user_message_id"]: job for job in generations}
    live = next((job["user_message_id"] for job in jobs.values() if job["status"] in ("queued", "running")), None)
    answers = {job["assistant_message_id"] for job in jobs.values() if job["assistant_message_id"]}
    by_id = {row["id"]: row for row in rows}
    result = []
    for row in rows:
        if row["id"] in answers:
            continue
        result.append(row)
        if job := jobs.get(row["id"]):
            if job["assistant_message_id"] in by_id:
                result.append(by_id[job["assistant_message_id"]])
            elif job["status"] != "completed":
                result.append({"role": "generation", "id": row["id"], "job": job, "connect": live == row["id"]})
    return result


def generation_bubble(request, session: dict, job: dict, oob=False, connect=None) -> str:
    if job["status"] == "completed":
        message = next((m for m in db.list_messages(session["id"]) if m["id"] == job["assistant_message_id"]), None)
        inner = templates.env.get_template("_message_bubble.html").render(
            request=request, session=session, message=message,
            is_last=message["id"] == (db.last_message(session["id"]) or {}).get("id"),
        ) if message else ""
    else:
        jobs = db.list_generations(session["id"])
        if connect is None:
            live = next((j["user_message_id"] for j in jobs if j["status"] in ("queued", "running")), None)
            connect = live == job["user_message_id"]
        latest_batch = jobs[-1]["batch_id"] if jobs else None
        inner = templates.env.get_template("_stream_bubble.html").render(
            request=request, session=session, job=job, connect=connect,
            retryable=job["batch_id"] == latest_batch,
        )
    if oob:
        return (inner or '<div></div>').replace('<div', f'<div hx-swap-oob="outerHTML:#stream-bubble-{job["user_message_id"]}"', 1)
    return inner


templates.env.globals["chat_context"] = chat_context
templates.env.globals["chat_messages"] = chat_messages
templates.env.globals["session_token_total"] = db.session_token_total
templates.env.globals["generation_bubble"] = generation_bubble
templates.env.globals["last_message"] = db.last_message


def folders_with_sessions() -> list[dict]:
    """Folders each carrying their sessions (for the sidebar)."""
    folders = db.list_folders()
    for folder in folders:
        folder["sessions"] = db.list_sessions(folder["id"])
    return folders


def folder_list_inner(request, current_folder_id=None, current_session_id=None) -> str:
    return templates.env.get_template("_folder_list.html").render(
        request=request,
        folders=folders_with_sessions(),
        current_folder_id=current_folder_id,
        current_session_id=current_session_id,
    )


def chat_surface_fragment(request, folder, session, oob=False) -> str:
    """Full `<div id="chat-surface">` fragment; `oob` marks it for out-of-band swap."""
    inner = templates.env.get_template("_chat_surface.html").render(
        request=request, folder=folder, session=session
    )
    attr = ' hx-swap-oob="true"' if oob else ""
    return f'<div id="chat-surface"{attr}>{inner}</div>'


def session_sidebar(request, session: dict) -> str:
    sidebar = folder_list_inner(request, session["folder_id"], session["id"])
    return f'<div id="folder-list" hx-swap-oob="innerHTML">{sidebar}</div>'
