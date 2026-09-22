"""Folder routes: CRUD + system-prompt partial."""

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

import db
import templating

router = APIRouter()


@router.post("/folders", response_class=HTMLResponse)
def create_folder(
    request: Request,
    name: str = Form(...),
    system_prompt: str = Form(""),
    active_session_id: int | None = Form(None),
):
    folder = db.create_folder(name.strip() or "New Folder", system_prompt)
    return templating.folder_list_inner(request, current_folder_id=folder["id"], current_session_id=active_session_id)


@router.post("/folders/{folder_id}/update", response_class=HTMLResponse)
def update_folder(
    request: Request,
    folder_id: int,
    name: str = Form(...),
    system_prompt: str = Form(""),
    chunk_limit: int | None = Form(None, ge=100, le=50000),
    active_session_id: int | None = Form(None),
):
    existing = db.get_folder(folder_id)
    if existing is None:
        raise HTTPException(404, "Folder not found.")
    if existing["type"] == "memo":
        system_prompt = existing["system_prompt"]
    db.update_folder(folder_id, name.strip() or "Untitled", system_prompt, chunk_limit if chunk_limit is not None else existing["chunk_limit"])
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
def delete_folder(request: Request, folder_id: int, active_session_id: int | None = Form(None)):
    if db.get_folder(folder_id) is None:
        raise HTTPException(404, "Folder not found.")
    current = db.get_session(active_session_id) if active_session_id else None
    db.delete_folder(folder_id)
    if current and current["folder_id"] != folder_id:
        return templating.folder_list_inner(request, current["folder_id"], current["id"])
    folders = db.list_folders()
    folder = folders[0] if folders else None
    sessions = db.list_sessions(folder["id"]) if folder else []
    session = sessions[0] if sessions else None
    return templating.folder_list_inner(request, folder["id"] if folder else None, session["id"] if session else None) + templating.chat_surface_fragment(request, folder, session, oob=True)


@router.post("/folders/{folder_id}/pin", response_class=HTMLResponse)
def toggle_pin(request: Request, folder_id: int, active_session_id: int | None = Form(None)):
    folder = db.get_folder(folder_id)
    if folder is None:
        return HTMLResponse("", status_code=404)
    db.set_folder_pin(folder_id, 0 if folder["pinned"] else 1)
    current = db.get_session(active_session_id) if active_session_id else None
    return templating.folder_list_inner(request, current_folder_id=current["folder_id"] if current else folder_id, current_session_id=active_session_id)


@router.post("/folders/{folder_id}/convert", response_class=HTMLResponse)
def convert_folder(
    request: Request,
    folder_id: int,
    folder_type: str = Form(...),
    active_session_id: int | None = Form(None),
):
    if folder_type not in ("chat", "memo", "translation"):
        return HTMLResponse("", status_code=400)
    if db.get_folder(folder_id) is None:
        raise HTTPException(404, "Folder not found.")
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
