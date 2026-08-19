"""SQLite storage for the chat app.

Single module-level connection guarded by a lock. FastAPI runs sync route
handlers in a threadpool, so every access function acquires the lock.
"""

import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  system_prompt TEXT NOT NULL DEFAULT '',
  is_memo INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
  title TEXT NOT NULL DEFAULT 'New Chat',
  model TEXT NOT NULL DEFAULT 'gemma4:31b',
  params_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  params_updated_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  reasoning TEXT,
  model TEXT,
  reasoning_effort TEXT,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  total_tokens INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

DEFAULT_PARAMS = {
    "reasoning_effort": "low",
    "temperature": 0.95,
    "top_p": 0.9,
    "max_tokens": 65535,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "seed": None,
}


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
    return _conn


def init_db() -> None:
    with _lock:
        conn = _get_conn()
        conn.executescript(SCHEMA)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "params_updated_at" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN params_updated_at TEXT")
        folder_cols = {row["name"] for row in conn.execute("PRAGMA table_info(folders)")}
        if "is_memo" not in folder_cols:
            conn.execute("ALTER TABLE folders ADD COLUMN is_memo INTEGER NOT NULL DEFAULT 0")
        if (
            conn.execute(
                "SELECT id FROM folders WHERE is_memo = 1 ORDER BY id LIMIT 1"
            ).fetchone()
            is None
        ):
            conn.execute(
                "INSERT INTO folders (name, system_prompt, is_memo) VALUES ('Memo', '', 1)"
            )
        msg_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if col not in msg_cols:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {col} INTEGER")
        conn.commit()


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


# --- folders -------------------------------------------------------------


def list_folders() -> list[dict]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT * FROM folders ORDER BY created_at, id"
        ).fetchall()
    return _rows_to_dicts(rows)


def get_folder(folder_id: int) -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()
    return dict(row) if row else None


def get_memo_folder() -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE is_memo = 1 ORDER BY id LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def create_folder(name: str, system_prompt: str = "") -> dict:
    with _lock:
        cur = _get_conn().execute(
            "INSERT INTO folders (name, system_prompt) VALUES (?, ?)",
            (name, system_prompt),
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def update_folder(folder_id: int, name: str, system_prompt: str) -> dict | None:
    with _lock:
        _get_conn().execute(
            "UPDATE folders SET name = ?, system_prompt = ? WHERE id = ?",
            (name, system_prompt, folder_id),
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_folder(folder_id: int) -> None:
    with _lock:
        _get_conn().execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        _get_conn().commit()


# --- sessions ------------------------------------------------------------


def list_sessions(folder_id: int) -> list[dict]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT * FROM sessions WHERE folder_id = ? ORDER BY updated_at DESC, id DESC",
            (folder_id,),
        ).fetchall()
    return _rows_to_dicts(rows)


def get_session(session_id: int) -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def model_usage() -> dict[str, int]:
    """Map model id -> number of sessions using it."""
    with _lock:
        rows = _get_conn().execute(
            "SELECT model, COUNT(*) AS n FROM sessions GROUP BY model"
        ).fetchall()
    return {row["model"]: row["n"] for row in rows}


def default_model() -> str:
    """Most-used model, or the schema default when no sessions exist."""
    usage = model_usage()
    if not usage:
        return "gemma4:31b"
    return max(usage, key=usage.get)


def last_params_session(folder_id: int) -> dict | None:
    """Most recent session in the folder whose params were edited, or None."""
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM sessions WHERE folder_id = ? AND params_updated_at IS NOT NULL"
            " ORDER BY params_updated_at DESC, id DESC LIMIT 1",
            (folder_id,),
        ).fetchone()
    return dict(row) if row else None


def create_session(
    folder_id: int, title: str = "New Chat", model: str | None = None,
    params_json: str | None = None,
) -> dict:
    if model is None or params_json is None:
        src = last_params_session(folder_id)
    if model is None:
        model = src["model"] if src is not None else default_model()
    if params_json is None:
        params_json = src["params_json"] if src is not None else json.dumps(DEFAULT_PARAMS)
    with _lock:
        cur = _get_conn().execute(
            "INSERT INTO sessions (folder_id, title, model, params_json) VALUES (?, ?, ?, ?)",
            (folder_id, title, model, params_json),
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def update_session(session_id: int, **fields) -> dict | None:
    if not fields:
        return get_session(session_id)
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _lock:
        _get_conn().execute(
            f"UPDATE sessions SET {cols}, updated_at = datetime('now') WHERE id = ?",
            (*fields.values(), session_id),
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def rename_session(session_id: int, title: str) -> dict | None:
    return update_session(session_id, title=title)


def delete_session(session_id: int) -> None:
    with _lock:
        _get_conn().execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        _get_conn().commit()


# --- messages ------------------------------------------------------------


def list_messages(session_id: int) -> list[dict]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return _rows_to_dicts(rows)


def add_message(
    session_id: int,
    role: str,
    content: str,
    reasoning: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> dict:
    with _lock:
        cur = _get_conn().execute(
            "INSERT INTO messages (session_id, role, content, reasoning, model,"
            " reasoning_effort, prompt_tokens, completion_tokens, total_tokens)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                reasoning,
                model,
                reasoning_effort,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ),
        )
        _get_conn().execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def session_token_total(session_id: int) -> dict:
    """Sum token usage across the session's assistant messages."""
    with _lock:
        row = _get_conn().execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0),"
            " COALESCE(SUM(completion_tokens), 0),"
            " COALESCE(SUM(total_tokens), 0)"
            " FROM messages WHERE session_id = ? AND role = 'assistant'",
            (session_id,),
        ).fetchone()
    return {
        "prompt": int(row[0]),
        "completion": int(row[1]),
        "total": int(row[2]),
    }


def delete_message(message_id: int) -> None:
    with _lock:
        _get_conn().execute("DELETE FROM messages WHERE id = ?", (message_id,))
        _get_conn().commit()


def set_system_message(session_id: int, content: str) -> None:
    """Mirror the folder's system prompt into exactly one system row.

    Updates in place when a row exists (keeps it the first message row);
    inserts otherwise. The folder row remains the source of truth.
    """
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND role = 'system'"
            " ORDER BY id LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?", (content, row["id"])
            )
        else:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, 'system', ?)",
                (session_id, content),
            )
        conn.commit()


def delete_system_message(session_id: int) -> None:
    with _lock:
        _get_conn().execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'system'",
            (session_id,),
        )
        _get_conn().commit()
