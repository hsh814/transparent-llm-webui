"""Folder routes: CRUD + system-prompt partial."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

import db
import templating

router = APIRouter()


@router.post("/folders", response_class=HTMLResponse)
def create_folder(
    request: Request,
    name: str = Form(...),
    system_prompt: str = Form(""),
):
    folder = db.create_folder(name.strip() or "New Folder", system_prompt)
    return templating.folder_list_inner(request, current_folder_id=folder["id"])


@router.post("/folders/{folder_id}/update", response_class=HTMLResponse)
def update_folder(
    request: Request,
    folder_id: int,
    name: str = Form(...),
    system_prompt: str = Form(""),
    chunk_limit: int | None = Form(None),
    active_session_id: int | None = Form(None),
):
    db.update_folder(folder_id, name.strip() or "Untitled", system_prompt, chunk_limit)
    folder = db.get_folder(folder_id)
    response = templating.folder_list_inner(
        request, current_folder_id=folder_id, current_session_id=active_session_id
    )
    if active_session_id is not None:
        session = db.get_session(active_session_id)
        if session is not None and session["folder_id"] == folder_id:
            response += templating.chat_surface_fragment(
                request, folder, session, oob=True
            )
    return response


@router.post("/folders/{folder_id}/delete", response_class=HTMLResponse)
def delete_folder(request: Request, folder_id: int):
    db.delete_folder(folder_id)
    return templating.folder_list_inner(request)


@router.post("/folders/{folder_id}/pin", response_class=HTMLResponse)
def toggle_pin(request: Request, folder_id: int):
    folder = db.get_folder(folder_id)
    if folder is None:
        return HTMLResponse("", status_code=404)
    db.set_folder_pin(folder_id, 0 if folder["pinned"] else 1)
    return templating.folder_list_inner(request, current_folder_id=folder_id)


@router.post("/folders/{folder_id}/convert", response_class=HTMLResponse)
def convert_folder(
    request: Request,
    folder_id: int,
    folder_type: str = Form(...),
    active_session_id: int | None = Form(None),
):
    if folder_type not in ("chat", "memo", "translation"):
        return HTMLResponse("", status_code=400)
    db.set_folder_type(folder_id, folder_type)
    folder = db.get_folder(folder_id)
    response = templating.folder_list_inner(
        request, current_folder_id=folder_id, current_session_id=active_session_id
    )
    if active_session_id is not None:
        session = db.get_session(active_session_id)
        if session is not None and session["folder_id"] == folder_id:
            response += templating.chat_surface_fragment(
                request, folder, session, oob=True
            )
    return response


@router.get("/folders/{folder_id}/system-prompt", response_class=HTMLResponse)
def system_prompt_partial(request: Request, folder_id: int):
    folder = db.get_folder(folder_id)
    if folder is None:
        return HTMLResponse("", status_code=404)
    return templating.templates.TemplateResponse(
        request, "_system_prompt.html", {"folder": folder}
    )
