"""Jinja2 templates + fragment helpers shared by app and routers."""

import json

from fastapi.templating import Jinja2Templates

import db
import ollama

templates = Jinja2Templates(directory="templates")


def _from_json(value: str) -> dict:
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


templates.env.filters["from_json"] = _from_json


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
        ctx_cap = ollama.model_context_length(session["model"]) or 8192
    except Exception:
        ctx_cap = 8192
    return {"models": models, "ctx_cap": ctx_cap, "usage": db.model_usage()}


def chat_messages(session_id: int) -> list[dict]:
    return db.list_messages(session_id)


templates.env.globals["chat_context"] = chat_context
templates.env.globals["chat_messages"] = chat_messages
templates.env.globals["session_token_total"] = db.session_token_total


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
