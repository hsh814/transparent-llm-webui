"""SQLite storage for the chat app.

Single module-level connection guarded by a lock. FastAPI runs sync route
handlers in a threadpool, so every access function acquires the lock.
"""

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(os.environ.get("CHAT_DB_PATH", Path(__file__).parent / "chat.db"))

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  system_prompt TEXT NOT NULL DEFAULT '',
  type TEXT NOT NULL DEFAULT 'chat',
  chunk_limit INTEGER,
  pinned INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
  title TEXT NOT NULL DEFAULT 'New Chat',
  auto_title INTEGER NOT NULL DEFAULT 1,
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

CREATE TABLE IF NOT EXISTS system_prompts (
  hash TEXT PRIMARY KEY,
  content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
  user_message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  params_json TEXT NOT NULL,
  messages_json TEXT NOT NULL,
  system_prompt_hash TEXT,
  batch_id INTEGER,
  attempt INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',
  content TEXT NOT NULL DEFAULT '',
  reasoning TEXT NOT NULL DEFAULT '',
  error TEXT,
  assistant_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS generations_session ON generations(session_id, user_message_id);
"""

DEFAULT_PARAMS = {
    "reasoning_effort": "low",
    "temperature": 0.95,
    "top_p": 0.9,
    "max_tokens": 65535,
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
        if "auto_title" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN auto_title INTEGER NOT NULL DEFAULT 1")
            for row in conn.execute("SELECT id, title FROM sessions").fetchall():
                conn.execute("UPDATE sessions SET auto_title = ? WHERE id = ?",
                             (_is_default_title(row["title"]), row["id"]))
        folder_cols = {row["name"] for row in conn.execute("PRAGMA table_info(folders)")}
        if "type" not in folder_cols:
            conn.execute("ALTER TABLE folders ADD COLUMN type TEXT NOT NULL DEFAULT 'chat'")
            if "is_memo" in folder_cols:
                conn.execute("UPDATE folders SET type = 'memo' WHERE is_memo = 1")
                conn.execute("ALTER TABLE folders DROP COLUMN is_memo")
        if "chunk_limit" not in folder_cols:
            conn.execute("ALTER TABLE folders ADD COLUMN chunk_limit INTEGER")
        if "pinned" not in folder_cols:
            conn.execute("ALTER TABLE folders ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        msg_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if col not in msg_cols:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {col} INTEGER")
        if "system_prompt_hash" not in msg_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN system_prompt_hash TEXT")
        # Backfill labels without making old conversations look newly modified.
        for row in conn.execute("SELECT id FROM sessions WHERE auto_title = 1").fetchall():
            _refresh_auto_title(conn, row["id"])
        conn.commit()


def recover_generations() -> None:
    """A process restart must leave interrupted work explicitly retryable."""
    with _lock, _get_conn() as conn:
        conn.execute(
            "UPDATE generations SET status = 'failed', error = 'The server restarted. Retry to continue.'"
            " WHERE status IN ('queued', 'running')"
        )


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


# --- folders -------------------------------------------------------------


def list_folders() -> list[dict]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT f.* FROM folders f"
            " LEFT JOIN (SELECT folder_id, MAX(updated_at) AS last_chat FROM sessions GROUP BY folder_id) s"
            " ON s.folder_id = f.id"
            " ORDER BY f.pinned DESC,"
            " COALESCE(s.last_chat, f.created_at) DESC, f.id DESC"
        ).fetchall()
    return _rows_to_dicts(rows)


def get_folder(folder_id: int) -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()
    return dict(row) if row else None


def create_folder(name: str, system_prompt: str = "", folder_type: str = "chat",
                  chunk_limit: int | None = None) -> dict:
    with _lock:
        cur = _get_conn().execute(
            "INSERT INTO folders (name, system_prompt, type, chunk_limit) VALUES (?, ?, ?, ?)",
            (name, system_prompt, folder_type, chunk_limit),
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def update_folder(folder_id: int, name: str, system_prompt: str,
                  chunk_limit: int | None = None) -> dict | None:
    with _lock:
        _get_conn().execute(
            "UPDATE folders SET name = ?, system_prompt = ?, chunk_limit = ? WHERE id = ?",
            (name, system_prompt, chunk_limit, folder_id),
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


def set_folder_pin(folder_id: int, pinned: int) -> dict | None:
    with _lock:
        _get_conn().execute(
            "UPDATE folders SET pinned = ? WHERE id = ?", (pinned, folder_id)
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()
    return dict(row) if row else None


def set_folder_type(folder_id: int, folder_type: str) -> dict | None:
    with _lock:
        _get_conn().execute(
            "UPDATE folders SET type = ? WHERE id = ?", (folder_type, folder_id)
        )
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()
    return dict(row) if row else None


# --- sessions ------------------------------------------------------------


def _is_default_title(title: str | None) -> bool:
    return not title or title.strip().casefold() in ("", "new chat")


def _refresh_auto_title(conn: sqlite3.Connection, session_id: int) -> None:
    """Derive a stable label from the first input; never overwrite a manual title."""
    session = conn.execute("SELECT auto_title FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if session is None or not session["auto_title"]:
        return
    first = conn.execute(
        "SELECT id, content FROM messages WHERE session_id = ? AND role IN ('user', 'memo') ORDER BY id LIMIT 1",
        (session_id,),
    ).fetchone()
    content = first["content"] if first else ""
    if first:
        # Translation chunks belong to one submission, even with tiny chunk limits.
        batch = conn.execute("SELECT batch_id FROM generations WHERE user_message_id = ?", (first["id"],)).fetchone()
        if batch and batch["batch_id"] is not None:
            content = "".join(row["content"] for row in conn.execute(
                "SELECT m.content FROM messages m JOIN generations g ON g.user_message_id = m.id"
                " WHERE g.session_id = ? AND g.batch_id = ? ORDER BY m.id", (session_id, batch["batch_id"])))
    title = " ".join(content.split())
    if len(title) > 60:
        title = title[:59].rstrip() + "…"
    title = title or f"New chat #{session_id}"
    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))


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
            "INSERT INTO sessions (folder_id, title, auto_title, model, params_json) VALUES (?, ?, ?, ?, ?)",
            (folder_id, title, _is_default_title(title), model, params_json),
        )
        _refresh_auto_title(_get_conn(), cur.lastrowid)
        _get_conn().commit()
        row = _get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def update_session(session_id: int, **fields) -> dict | None:
    if not fields:
        return get_session(session_id)
    if fields.keys() - {"title", "model", "params_json", "params_updated_at"}:
        raise ValueError("Unsupported session field")
    if "title" in fields:
        fields["title"] = fields["title"].strip()
        fields["auto_title"] = not fields["title"]
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _lock:
        _get_conn().execute(
            f"UPDATE sessions SET {cols}, updated_at = datetime('now') WHERE id = ?",
            (*fields.values(), session_id),
        )
        if "title" in fields:
            _refresh_auto_title(_get_conn(), session_id)
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


def next_user_message(session_id: int, after_message_id: int) -> dict | None:
    """Next role='user' message by ID after the given one (used by translation chain)."""
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM messages WHERE session_id = ? AND role = 'user' AND id > ?"
            " ORDER BY id LIMIT 1",
            (session_id, after_message_id),
        ).fetchone()
    return dict(row) if row else None


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
    system_prompt_hash: str | None = None,
) -> dict:
    with _lock:
        cur = _get_conn().execute(
            "INSERT INTO messages (session_id, role, content, reasoning, model,"
            " reasoning_effort, prompt_tokens, completion_tokens, total_tokens,"
            " system_prompt_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                system_prompt_hash,
            ),
        )
        _get_conn().execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        _refresh_auto_title(_get_conn(), session_id)
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


def upsert_system_prompt(content: str) -> str:
    """Content-address the prompt text; returns its sha256 hex (idempotent)."""
    hash_ = hashlib.sha256(content.encode()).hexdigest()
    with _lock:
        _get_conn().execute(
            "INSERT OR IGNORE INTO system_prompts (hash, content) VALUES (?, ?)",
            (hash_, content),
        )
        _get_conn().commit()
    return hash_


def get_system_prompt_by_hash(hash_: str) -> str | None:
    """Look up the verbatim prompt for a hash; None when missing."""
    with _lock:
        row = _get_conn().execute(
            "SELECT content FROM system_prompts WHERE hash = ?", (hash_,)
        ).fetchone()
    return row["content"] if row else None


def last_message(session_id: int) -> dict | None:
    """Highest-id message row in the session (the only deletable one)."""
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


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


class GenerationBusy(ValueError):
    pass


def enqueue_turns(session_id: int, chunks: list[str]) -> list[dict]:
    """Atomically save input and the exact request, before any network work."""
    with _lock, _get_conn() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise ValueError("Chat not found")
        folder = conn.execute("SELECT * FROM folders WHERE id = ?", (session["folder_id"],)).fetchone()
        if folder["type"] == "memo":
            raise ValueError("This folder stores notes")
        if conn.execute(
            "SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')",
            (session_id,),
        ).fetchone():
            raise GenerationBusy("Wait for the current response or stop it before sending another message.")
        prompt = folder["system_prompt"]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        conn.execute("INSERT OR IGNORE INTO system_prompts VALUES (?, ?)", (prompt_hash, prompt))
        history = [dict(row) for row in conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user', 'assistant') ORDER BY id",
            (session_id,),
        )]
        prefix = [{"role": "system", "content": prompt}] if prompt else []
        ids = []
        for chunk in chunks:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, system_prompt_hash) VALUES (?, 'user', ?, ?)",
                (session_id, chunk, prompt_hash),
            )
            user_id = cur.lastrowid
            messages = prefix + (history if folder["type"] == "chat" else []) + [{"role": "user", "content": chunk}]
            conn.execute(
                "INSERT INTO generations (user_message_id, session_id, model, params_json, messages_json, system_prompt_hash, batch_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, session["model"],
                 json.dumps({k: v for k, v in json.loads(session["params_json"]).items() if k in DEFAULT_PARAMS}),
                 json.dumps(messages), prompt_hash, ids[0] if ids else user_id),
            )
            ids.append(user_id)
        _refresh_auto_title(conn, session_id)
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))
        return [dict(conn.execute("SELECT * FROM messages WHERE id = ?", (uid,)).fetchone()) for uid in ids]


def list_generations(session_id: int) -> list[dict]:
    with _lock:
        return _rows_to_dicts(_get_conn().execute(
            "SELECT * FROM generations WHERE session_id = ? ORDER BY user_message_id", (session_id,)
        ).fetchall())


def conversation_snapshot(session_id: int) -> tuple[list[dict], list[dict]]:
    """Read replies and generation states together while workers are active."""
    with _lock:
        conn = _get_conn()
        rows = _rows_to_dicts(conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall())
        jobs = _rows_to_dicts(conn.execute("SELECT * FROM generations WHERE session_id = ? ORDER BY user_message_id", (session_id,)).fetchall())
        return rows, jobs


def delete_last_message(session_id: int, message_id: int) -> bool:
    with _lock, _get_conn() as conn:
        if conn.execute("SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')", (session_id,)).fetchone():
            raise GenerationBusy("Stop the response before deleting messages.")
        row = conn.execute("SELECT MAX(id) FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        if row[0] != message_id:
            return False
        conn.execute("DELETE FROM messages WHERE id = ? AND session_id = ?", (message_id, session_id))
        _refresh_auto_title(conn, session_id)
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))
        return True


def get_generation(session_id: int, user_message_id: int) -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM generations WHERE session_id = ? AND user_message_id = ?",
            (session_id, user_message_id),
        ).fetchone()
        return dict(row) if row else None


def claim_generation(session_id: int) -> dict | None:
    with _lock, _get_conn() as conn:
        if conn.execute("SELECT 1 FROM generations WHERE session_id = ? AND status = 'running'", (session_id,)).fetchone():
            return None
        row = conn.execute(
            "SELECT * FROM generations WHERE session_id = ? AND status = 'queued' ORDER BY user_message_id LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE generations SET status = 'running' WHERE user_message_id = ?", (row["user_message_id"],))
        return dict(row)


def update_generation(job: dict, content: str, reasoning: str) -> bool:
    with _lock, _get_conn() as conn:
        return conn.execute(
            "UPDATE generations SET content = ?, reasoning = ? WHERE user_message_id = ? AND attempt = ? AND status = 'running'",
            (content, reasoning, job["user_message_id"], job["attempt"]),
        ).rowcount == 1


def finish_generation(job: dict, content: str, reasoning: str, usage: dict | None) -> None:
    usage = usage or {}
    with _lock, _get_conn() as conn:
        row = conn.execute("SELECT status, attempt FROM generations WHERE user_message_id = ?", (job["user_message_id"],)).fetchone()
        if row is None or row["status"] != "running" or row["attempt"] != job["attempt"]:
            return
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, reasoning, model, reasoning_effort,"
            " prompt_tokens, completion_tokens, total_tokens, system_prompt_hash)"
            " VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)",
            (job["session_id"], content, reasoning or None, job["model"],
             json.loads(job["params_json"]).get("reasoning_effort"), usage.get("prompt_tokens"),
             usage.get("completion_tokens"), usage.get("total_tokens"), job["system_prompt_hash"]),
        )
        conn.execute(
            "UPDATE generations SET status = 'completed', content = ?, reasoning = ?, assistant_message_id = ?"
            " WHERE user_message_id = ?", (content, reasoning, cur.lastrowid, job["user_message_id"]),
        )
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (job["session_id"],))


def interrupt_generations(session_id: int, error: str, status: str = "failed", job: dict | None = None) -> None:
    with _lock, _get_conn() as conn:
        if job is not None:
            current = conn.execute("SELECT status, attempt FROM generations WHERE user_message_id = ?", (job["user_message_id"],)).fetchone()
            if current is None or current["status"] != "running" or current["attempt"] != job["attempt"]:
                return
        changed = conn.execute(
            "UPDATE generations SET status = ?, error = ? WHERE session_id = ? AND status IN ('queued', 'running')",
            (status, error, session_id),
        ).rowcount
        if changed:
            conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))


def retry_generations(session_id: int) -> bool:
    with _lock, _get_conn() as conn:
        if conn.execute(
            "SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')", (session_id,)
        ).fetchone():
            raise GenerationBusy("A response is already in progress.")
        # Only retry the latest submission: old failures must never be silently replayed.
        latest = conn.execute("SELECT MAX(id) FROM messages WHERE session_id = ? AND role = 'user'", (session_id,)).fetchone()[0]
        last_job = conn.execute("SELECT status, batch_id FROM generations WHERE user_message_id = ?", (latest,)).fetchone()
        if last_job is None or last_job["status"] not in ("failed", "cancelled"):
            return False
        conn.execute(
            "UPDATE generations SET status = 'queued', content = '', reasoning = '', error = NULL, attempt = attempt + 1"
            " WHERE session_id = ? AND batch_id = ? AND status IN ('failed', 'cancelled')",
            (session_id, last_job["batch_id"]),
        )
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))
        return True
