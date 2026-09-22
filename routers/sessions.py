"""Session routes: create, rename, delete, model/params, chat surface."""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import db
import ollama
import templating

router = APIRouter()


@router.post("/start", response_class=HTMLResponse)
def quick_start(request: Request, active_session_id: int | None = Form(None)):
    current = db.get_session(active_session_id) if active_session_id else None
    folders = db.list_folders()
    folder_id = current["folder_id"] if current else (folders[0]["id"] if folders else db.create_folder("Personal")["id"])
    return create_session(request, folder_id)


@router.post("/folders/{folder_id}/sessions", response_class=HTMLResponse)
def create_session(request: Request, folder_id: int):
    folder = db.get_folder(folder_id)
    if folder is None:
        return HTMLResponse("", status_code=404)
    session = db.create_session(folder_id)
    return templating.folder_list_inner(
        request, current_folder_id=folder_id, current_session_id=session["id"]
    ) + templating.chat_surface_fragment(request, folder, session, oob=True)


@router.post("/sessions/{session_id}/rename", response_class=HTMLResponse)
def rename_session(request: Request, session_id: int, title: str = Form(...)):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    db.rename_session(session_id, title.strip() or "New Chat")
    return templating.folder_list_inner(
        request, current_folder_id=session["folder_id"], current_session_id=session_id
    )


@router.post("/sessions/{session_id}/delete", response_class=HTMLResponse)
def delete_session(request: Request, session_id: int, active_session_id: int | None = Form(None)):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    db.delete_session(session_id)
    current = db.get_session(active_session_id) if active_session_id else None
    if current is None:
        remaining = db.list_sessions(folder["id"])
        current = remaining[0] if remaining else None
    current_folder = db.get_folder(current["folder_id"]) if current else folder
    response = templating.folder_list_inner(request, current_folder_id=current_folder["id"], current_session_id=current["id"] if current else None)
    if active_session_id == session_id or active_session_id is None:
        response += templating.chat_surface_fragment(request, current_folder, current, oob=True)
    return response


@router.post("/sessions/{session_id}/model", response_class=HTMLResponse)
def update_model_params(
    request: Request,
    session_id: int,
    model: str = Form(..., min_length=1, max_length=200),
    reasoning_effort: str | None = Form(None, pattern="^(none|low|medium|high|max)$"),
    temperature: float | None = Form(None, ge=0, le=2),
    top_p: float | None = Form(None, ge=0, le=1),
    max_tokens: int | None = Form(None, ge=1),
    num_ctx: int | None = Form(None, ge=512),
    top_k: int | None = Form(None, ge=0, le=200),
    repeat_penalty: float | None = Form(None, ge=0, le=2),
    seed: str | None = Form(None),
):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    if not model.strip():
        raise HTTPException(422, "Choose a model.")
    if any(value is not None for value in (num_ctx, top_k, repeat_penalty)):
        raise HTTPException(422, "Context size, top K, and repeat penalty are not supported by this API endpoint.")
    params = {**db.DEFAULT_PARAMS, **json.loads(session["params_json"])}
    submitted = {
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    params.update({key: value for key, value in submitted.items() if value is not None})
    if seed is not None or request.headers.get("HX-Target") == "params-panel":
        try:
            params["seed"] = int(seed) if seed and seed.strip() else None
        except ValueError as exc:
            raise HTTPException(422, "Seed must be a whole number or blank.") from exc
    db.update_session(
        session_id, model=model, params_json=json.dumps(params),
        params_updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    session = db.get_session(session_id)
    ctx = templating.chat_context(session)
    models, ctx_cap = ctx["models"], ctx["ctx_cap"]
    selector = templating.templates.env.get_template("_model_selector.html").render(
        session=session, models=models, params=params, usage=db.model_usage(),
        ctx_cap=ctx_cap,
    )
    panel = templating.templates.env.get_template("_params_panel.html").render(
        session=session, params=params, ctx_cap=ctx_cap
    )
    if request.headers.get("HX-Target") == "params-panel":
        selector = selector.replace('id="model-selector"', 'id="model-selector" hx-swap-oob="true"', 1)
        return HTMLResponse(panel + selector, headers={"HX-Trigger": "settingsSaved"})
    panel = panel.replace('id="params-panel"', 'id="params-panel" hx-swap-oob="true"', 1)
    return HTMLResponse(selector + panel, headers={"HX-Trigger": "settingsSaved"})


@router.post("/sessions/{session_id}/model/refresh", response_class=HTMLResponse)
def refresh_model_cache(request: Request, session_id: int):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    ollama.clear_cache()
    ctx = templating.chat_context(session)
    models, ctx_cap = ctx["models"], ctx["ctx_cap"]
    params = json.loads(session["params_json"] or "{}")
    selector = templating.templates.env.get_template("_model_selector.html").render(
        session=session, models=models, params=params, usage=db.model_usage(),
        ctx_cap=ctx_cap,
    )
    panel = templating.templates.env.get_template("_params_panel.html").render(
        session=session, params=params, ctx_cap=ctx_cap
    )
    panel = panel.replace(
        '<form class="params-panel" id="params-panel"',
        '<form class="params-panel" id="params-panel" hx-swap-oob="true"',
        1,
    )
    return HTMLResponse(selector + panel)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_surface(request: Request, session_id: int):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    if request.headers.get("HX-Request") != "true":
        return RedirectResponse(f"/?session={session_id}", status_code=303)
    folder = db.get_folder(session["folder_id"])
    sidebar = templating.folder_list_inner(
        request,
        current_folder_id=session["folder_id"],
        current_session_id=session_id,
    )
    return templating.chat_surface_fragment(
        request, folder, session
    ) + f'<div id="folder-list" hx-swap-oob="innerHTML">{sidebar}</div>'
