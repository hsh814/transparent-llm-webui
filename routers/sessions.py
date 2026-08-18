"""Session routes: create, rename, delete, model/params, chat surface."""

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

import db
import ollama
import templating

router = APIRouter()


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
def delete_session(request: Request, session_id: int):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    db.delete_session(session_id)
    return templating.folder_list_inner(request, current_folder_id=folder["id"])


@router.post("/sessions/{session_id}/model", response_class=HTMLResponse)
def update_model_params(
    request: Request,
    session_id: int,
    model: str = Form(...),
    reasoning_effort: str = Form("low"),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    max_tokens: int = Form(1024),
    num_ctx: int = Form(8192),
    top_k: int = Form(40),
    repeat_penalty: float = Form(1.1),
    seed: str = Form(""),
):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    params = {
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "num_ctx": num_ctx,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "seed": int(seed) if seed.strip() else None,
    }
    db.update_session(session_id, model=model, params_json=json.dumps(params))
    session = db.get_session(session_id)
    try:
        models = ollama.list_models()
    except Exception:
        models = [model]
    try:
        ctx_cap = ollama.model_context_length(model) or 8192
    except Exception:
        ctx_cap = 8192
    selector = templating.templates.env.get_template("_model_selector.html").render(
        session=session, models=models, params=params, usage=db.model_usage()
    )
    panel = templating.templates.env.get_template("_params_panel.html").render(
        session=session, params=params, ctx_cap=ctx_cap
    )
    return HTMLResponse(selector + panel)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_surface(request: Request, session_id: int):
    session = db.get_session(session_id)
    if session is None:
        return HTMLResponse("", status_code=404)
    folder = db.get_folder(session["folder_id"])
    sidebar = templating.folder_list_inner(
        request,
        current_folder_id=session["folder_id"],
        current_session_id=session_id,
    )
    return templating.chat_surface_fragment(
        request, folder, session
    ) + f'<div id="folder-list" hx-swap-oob="true">{sidebar}</div>'
