"""Transparent Chat — FastAPI app.

Serves the htmx UI and a small JSON-free HTML/SSE API. Single user, no auth.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before local imports read env (ollama.py)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import db
import templating
from routers import chat, folders, sessions

if not os.environ.get("OLLAMA_API_KEY"):
    raise RuntimeError(
        "OLLAMA_API_KEY is not set. Add it to .env (or export it) before starting."
    )

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    db.recover_generations()
    yield


app = FastAPI(title="Transparent Chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(folders.router)
app.include_router(sessions.router)
app.include_router(chat.router)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, session: int | None = None):
    folders = templating.folders_with_sessions()
    folder = folders[0] if folders else None
    selected = db.get_session(session) if session is not None else None
    if selected:
        folder = db.get_folder(selected["folder_id"])
    elif folder is not None:
        sessions_ = folder["sessions"]
        selected = sessions_[0] if sessions_ else None
    return templating.templates.TemplateResponse(
        request, "index.html",
        {
            "request": request,
            "folders": folders,
            "current_folder": folder,
            "current_session": selected,
        },
    )
