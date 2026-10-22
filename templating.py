"""Jinja2 templates + fragment helpers shared by app and routers."""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

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
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


templates.env.filters["from_json"] = _from_json


def _http_url(value) -> str:
    """Only web source URLs may become clickable links."""
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value):
        return ""
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme.lower() in ("http", "https") and parsed.hostname else ""
    except ValueError:
        return ""


def _image_url(value) -> str:
    if isinstance(value, str) and value.startswith((
        "data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,"
    )):
        return value
    return ""


templates.env.filters["http_url"] = _http_url
templates.env.filters["image_url"] = _image_url


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


def message_controls(session: dict, message: dict, *, rows=None, jobs=None) -> dict:
    """Small action metadata, also used by append-only message fragments."""
    if rows is None:
        rows, jobs = db.conversation_snapshot(session["id"])
    jobs = jobs or []
    model_conversation = db.get_folder(session["folder_id"])["type"] != "memo"
    latest = next((row for row in reversed(rows) if row["role"] != "system"), None)
    return {
        "is_last": bool(latest and latest["id"] == message["id"]),
        "can_respond": model_conversation and message["role"] == "user",
        "has_trace": any(job["user_message_id"] == message["id"] and job.get("trace") for job in jobs),
        "can_regenerate": model_conversation and message["role"] == "assistant"
        and (bool(message.get("user_message_id")) or not message.get("provenance")),
        "active": any(job["status"] in ("queued", "running") for job in jobs),
        "variants": db.response_variant_state(session["id"], message["user_message_id"])
        if message.get("user_message_id") else None,
    }


def chat_messages(session_id: int) -> list[dict]:
    rows, generations = db.conversation_snapshot(session_id)
    session = db.get_session(session_id)
    jobs = {job["user_message_id"]: job for job in generations}
    live = next((job["user_message_id"] for job in generations if job["status"] in ("queued", "running")), None)
    answers = {job["assistant_message_id"] for job in generations if job["assistant_message_id"]}
    result = []
    for row in rows:
        if row["id"] in answers:
            continue
        job = jobs.get(row["id"])
        result.append({**row, "job": job, "ui": message_controls(session, row, rows=rows, jobs=generations)})
        if job:
            result.append({"role": "generation", "id": row["id"], "job": job, "connect": live == row["id"]})
    return result


def trace_timeline(session: dict, job: dict, *, busy: bool = False) -> str:
    return templates.env.get_template("_tool_timeline.html").render(
        session=session, trace_job=job, trace=job.get("trace", []), busy=busy,
    )


def generation_bubble(request, session: dict, job: dict, oob=False, connect=None) -> str:
    rows, jobs = db.conversation_snapshot(session["id"])
    message = next((row for row in rows if row["id"] == job["assistant_message_id"]), None)
    active = job["status"] in ("queued", "running")
    if connect is None:
        live = next((j["user_message_id"] for j in jobs if j["status"] in ("queued", "running")), None)
        connect = live == job["user_message_id"]
    if message:
        message = {**message, "user_message_id": job["user_message_id"]}
        message["ui"] = message_controls(session, message, rows=rows, jobs=jobs)
    inner = templates.env.get_template("_stream_bubble.html").render(
        request=request, session=session, job=job, message=message,
        response_job=db.get_message_generation(session["id"], message["id"]) if message else None,
        connect=bool(active and connect), active=active,
        busy=any(j["status"] in ("queued", "running") for j in jobs),
        retryable=bool(jobs and job["batch_id"] == jobs[-1]["batch_id"]),
        oob=oob,
    )
    return inner


templates.env.globals["chat_context"] = chat_context
templates.env.globals["chat_messages"] = chat_messages
templates.env.globals["session_token_total"] = db.session_token_total
templates.env.globals["generation_bubble"] = generation_bubble
templates.env.globals["message_controls"] = message_controls
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


def chat_surface_fragment(request, folder, session, oob=False, *, editor_web_search=None) -> str:
    """Full `<div id="chat-surface">` fragment; `oob` marks it for out-of-band swap."""
    inner = templates.env.get_template("_chat_surface.html").render(
        request=request, folder=folder, session=session, editor_web_search=editor_web_search,
    )
    attr = ' hx-swap-oob="true"' if oob else ""
    return f'<div id="chat-surface"{attr}>{inner}</div>'


def session_sidebar(request, session: dict) -> str:
    sidebar = folder_list_inner(request, session["folder_id"], session["id"])
    return f'<div id="folder-list" hx-swap-oob="innerHTML">{sidebar}</div>'
