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
from datetime import datetime, timezone

import inputs
import payloads
from tool_events import allowed_tools, api_messages, pending_calls, round_request

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
  attachments_id TEXT REFERENCES payloads(id),
  api_content_id TEXT REFERENCES payloads(id),
  provenance_id TEXT REFERENCES payloads(id),
  system_prompt_hash TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS system_prompts (
  hash TEXT PRIMARY KEY,
  content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
  user_message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  request_id TEXT NOT NULL REFERENCES payloads(id),
  base_request_id TEXT REFERENCES payloads(id),
  trace_id TEXT REFERENCES payloads(id),
  batch_id INTEGER,
  attempt INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',
  content TEXT NOT NULL DEFAULT '',
  reasoning TEXT NOT NULL DEFAULT '',
  error TEXT,
  assistant_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS response_variants (
  assistant_message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  user_message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  request_id TEXT REFERENCES payloads(id),
  base_request_id TEXT REFERENCES payloads(id),
  trace_id TEXT REFERENCES payloads(id)
);
CREATE INDEX IF NOT EXISTS variants_user ON response_variants(user_message_id, assistant_message_id);
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


def _request_from_legacy(row: dict) -> dict:
    """Discard mutable lifecycle data from an old generation or variant snapshot."""
    return {
        "model": row["model"],
        "params": json.loads(row["params_json"]),
        "messages": json.loads(row["messages_json"]),
        "search": json.loads(row["search_json"]) if row.get("search_json") is not None else None,
        "system_prompt_hash": row.get("system_prompt_hash"),
    }


def _migrate_payloads(conn: sqlite3.Connection) -> None:
    message_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    for name in ("attachments_id", "api_content_id", "provenance_id"):
        if name not in message_cols:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {name} TEXT REFERENCES payloads(id)")
    if "attachments_json" in message_cols or "api_content_json" in message_cols:
        for row in conn.execute("SELECT * FROM messages"):
            attachments = json.loads(row["attachments_json"]) if "attachments_json" in message_cols else []
            api_content = (json.loads(row["api_content_json"])
                           if "api_content_json" in message_cols and row["api_content_json"] is not None else None)
            conn.execute(
                "UPDATE messages SET attachments_id = ?, api_content_id = ? WHERE id = ?",
                (payloads.store(conn, attachments),
                 payloads.store(conn, api_content) if api_content is not None else None, row["id"]),
            )
        for name in ("attachments_json", "api_content_json"):
            if name in message_cols:
                conn.execute(f"ALTER TABLE messages DROP COLUMN {name}")
    generation_cols = {row["name"] for row in conn.execute("PRAGMA table_info(generations)")}
    if "request_id" not in generation_cols:
        if "search_json" not in generation_cols:
            conn.execute("ALTER TABLE generations ADD COLUMN search_json TEXT")
        if "system_prompt_hash" not in generation_cols:
            conn.execute("ALTER TABLE generations ADD COLUMN system_prompt_hash TEXT")
        generation_schema = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS generations"):SCHEMA.index("CREATE TABLE IF NOT EXISTS response_variants")]
        conn.execute(generation_schema.replace("IF NOT EXISTS generations", "generations_payload_migration").strip().rstrip(";"))
        columns = ("user_message_id", "session_id", "request_id", "batch_id", "attempt",
                   "status", "content", "reasoning", "error", "assistant_message_id")
        for row in conn.execute("SELECT * FROM generations"):
            old = dict(row)
            old["request_id"] = payloads.store(conn, _request_from_legacy(old))
            conn.execute(
                f"INSERT INTO generations_payload_migration ({', '.join(columns)})"
                f" VALUES ({', '.join('?' for _ in columns)})",
                tuple(old[name] for name in columns),
            )
        conn.execute("DROP TABLE generations")
        conn.execute("ALTER TABLE generations_payload_migration RENAME TO generations")
        conn.execute("CREATE INDEX generations_session ON generations(session_id, user_message_id)")
    variant_cols = {row["name"] for row in conn.execute("PRAGMA table_info(response_variants)")}
    if "request_id" not in variant_cols:
        conn.execute("ALTER TABLE response_variants ADD COLUMN request_id TEXT REFERENCES payloads(id)")
    if "request_json" in variant_cols:
        for row in conn.execute("SELECT * FROM response_variants WHERE request_json IS NOT NULL"):
            request_id = payloads.store(conn, _request_from_legacy(json.loads(row["request_json"])))
            conn.execute("UPDATE response_variants SET request_id = ? WHERE assistant_message_id = ?",
                         (request_id, row["assistant_message_id"]))
        conn.execute("ALTER TABLE response_variants DROP COLUMN request_json")


def _legacy_trace(search: dict | None, request: dict, user_id: int) -> list[dict]:
    if not search:
        return []
    call_id = f"legacy-search-{user_id}"
    arguments = json.dumps({"query": search.get("query", ""),
                            "max_results": search.get("max_results", 5)}, ensure_ascii=False)
    call = {"id": call_id, "type": "function",
            "function": {"name": "web_search", "arguments": arguments}}
    completed = search.get("status") == "completed"
    result = ({key: search.get(key) for key in ("query", "max_results", "results", "fetched_at")}
              if completed else {"error": search.get("error") or
                                 "Legacy app search was not completed. Regenerate to choose new searches."})
    return [
        {"kind": "assistant", "status": "completed", "origin": "legacy",
         "message": {"role": "assistant", "content": "", "tool_calls": [call]},
         "reasoning": "", "request": request, "usage": None},
        {"kind": "tool", "status": "completed" if completed else "failed", "origin": "legacy",
         "call_id": call_id, "name": "web_search", "arguments": arguments,
         "started_at": search.get("fetched_at") or "",
         "message": {"role": "tool", "tool_call_id": call_id, "name": "web_search",
                     "content": json.dumps(result, ensure_ascii=False)},
         **({} if completed else {"error": result["error"]})},
    ]


def _clean_base(conn: sqlite3.Connection, user_id: int, request: dict) -> dict:
    """Rebuild inputs from raw rows, never from search delimiters in user text."""
    user = conn.execute("SELECT session_id FROM messages WHERE id = ?", (user_id,)).fetchone()
    folder_type = conn.execute(
        "SELECT f.type FROM folders f JOIN sessions s ON s.folder_id = f.id WHERE s.id = ?",
        (user["session_id"],),
    ).fetchone()[0]
    rows = _visible_messages(conn, user["session_id"])
    prefix = []
    for row in rows:
        if row["role"] in ("user", "assistant"):
            if folder_type == "chat" or row["id"] == user_id:
                prefix.append(row)
        if row["id"] == user_id:
            break
    return {**request,
            "messages": [message for message in request["messages"] if message["role"] == "system"]
                        + _history_messages(conn, prefix, exclude_trace_user_id=user_id)}


def _migrate_tool_traces(conn: sqlite3.Connection) -> None:
    """Upgrade every independent snapshot atomically; NULL means still unknown."""
    for table in ("generations", "response_variants"):
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name in ("base_request_id", "trace_id"):
            if name not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT REFERENCES payloads(id)")
    rebuild_all_users = conn.execute("PRAGMA user_version").fetchone()[0] < 1
    users = conn.execute(
        "SELECT m.id, m.content, m.attachments_id FROM messages m WHERE m.role = 'user'"
        + ("" if rebuild_all_users else
           " AND (EXISTS (SELECT 1 FROM generations g WHERE g.user_message_id = m.id"
           " AND g.base_request_id IS NULL) OR EXISTS"
           " (SELECT 1 FROM response_variants v WHERE v.user_message_id = m.id"
           " AND v.request_id IS NOT NULL AND v.base_request_id IS NULL))")
    )
    for row in users:
        store = payloads.Store(conn)
        attachments = store.get(row["attachments_id"]) if row["attachments_id"] else []
        api_content_id = store.put(inputs.build_content(row["content"], attachments))
        conn.execute("UPDATE messages SET api_content_id = ? WHERE id = ?", (api_content_id, row["id"]))
    for generation in conn.execute(
        "SELECT g.* FROM generations g WHERE g.base_request_id IS NULL OR EXISTS"
        " (SELECT 1 FROM response_variants v WHERE v.user_message_id = g.user_message_id"
        " AND v.request_id IS NOT NULL AND v.base_request_id IS NULL) ORDER BY g.user_message_id"
    ):
        user_id = generation["user_message_id"]
        for table, key, records in (
            ("response_variants", "assistant_message_id", conn.execute(
                "SELECT * FROM response_variants WHERE user_message_id = ?"
                " AND request_id IS NOT NULL AND base_request_id IS NULL ORDER BY assistant_message_id",
                (user_id,))),
            ("generations", "user_message_id", (generation,)),
        ):
            for row in records:
                if row["request_id"] is None or row["base_request_id"] is not None:
                    continue
                store = payloads.Store(conn)
                old = store.get(row["request_id"])
                search = old.get("search")
                request = {k: v for k, v in old.items() if k != "search"}
                request["tools"] = request.get("tools", [])
                base = _clean_base(conn, user_id, request)
                trace = _legacy_trace(search, request, user_id)
                conn.execute(
                    f"UPDATE {table} SET request_id = ?, base_request_id = ?, trace_id = ? WHERE {key} = ?",
                    (store.put(request), store.put(base), store.put(trace), row[key]),
                )
    if rebuild_all_users:
        conn.execute("PRAGMA user_version = 1")

def init_db() -> None:
    with _lock:
        conn = _get_conn()
        # Table replacement must not cascade into historical messages. DDL, node
        # conversion, backfill and FK validation all commit or roll back together.
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            for statement in (payloads.SCHEMA + SCHEMA).split(";"):
                if statement.strip():
                    conn.execute(statement)
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
            _migrate_payloads(conn)
            # Retain explicit legacy NULL snapshots; only linked answers without
            # any version record receive the current immutable request reference.
            conn.execute(
                "INSERT OR IGNORE INTO response_variants"
                " (assistant_message_id, user_message_id, request_id)"
                " SELECT assistant_message_id, user_message_id, request_id FROM generations"
                " WHERE assistant_message_id IS NOT NULL"
            )
            _migrate_tool_traces(conn)
            for row in conn.execute("SELECT id FROM sessions WHERE auto_title = 1").fetchall():
                _refresh_auto_title(conn, row["id"])
            payloads.collect(conn)
            if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise sqlite3.IntegrityError("Foreign key violation during payload migration")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")


def recover_generations() -> None:
    """A process restart must leave interrupted work explicitly retryable."""
    with _lock, _get_conn() as conn:
        for row in conn.execute("SELECT * FROM generations WHERE status IN ('queued', 'running')"):
            _fail_pending_tools(conn, row, "The server restarted. Retry to continue.")
        conn.execute(
            "UPDATE generations SET status = 'failed', error = 'The server restarted. Retry to continue.'"
            " WHERE status IN ('queued', 'running')"
        )


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


def _message(row: sqlite3.Row | dict, store: payloads.Store) -> dict:
    result = dict(row)
    result["attachments"] = store.get(result["attachments_id"]) if result["attachments_id"] else []
    result["api_content"] = store.get(result["api_content_id"]) if result["api_content_id"] else None
    result["provenance"] = store.get(result["provenance_id"]) if result.get("provenance_id") else None
    return result


def _generation(row: sqlite3.Row | dict, store: payloads.Store) -> dict:
    result = dict(row)
    request = store.get(result["request_id"])
    result.update(request)
    result["request"] = request
    result["base_request"] = (store.get(result["base_request_id"])
                              if result.get("base_request_id") else None)
    result["trace"] = store.get(result["trace_id"]) if result.get("trace_id") else []
    return result


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
    with _lock, _get_conn() as conn:
        conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        payloads.collect(conn)


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
    with _lock, _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        payloads.collect(conn)


# --- messages ------------------------------------------------------------


def list_messages(session_id: int) -> list[dict]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        store = payloads.Store(_get_conn())
        return [_message(row, store) for row in rows]


def _visible_messages(conn: sqlite3.Connection, session_id: int, *, decode: bool = True) -> list[dict]:
    """Pair selected answers with their inputs, even for prequeued translations."""
    rows = conn.execute(
        "SELECT m.* FROM messages m"
        " LEFT JOIN response_variants v ON v.assistant_message_id = m.id"
        " LEFT JOIN generations g ON g.user_message_id = v.user_message_id"
        " WHERE m.session_id = ? AND"
        " (v.assistant_message_id IS NULL OR g.assistant_message_id = m.id)"
        " ORDER BY COALESCE(v.user_message_id, m.id),"
        " CASE WHEN v.assistant_message_id IS NULL THEN 0 ELSE 1 END, m.id",
        (session_id,),
    ).fetchall()
    if not decode:
        return _rows_to_dicts(rows)
    store = payloads.Store(conn)
    return [_message(row, store) for row in rows]


def visible_messages(session_id: int) -> list[dict]:
    with _lock:
        return _visible_messages(_get_conn(), session_id)


def next_user_message(session_id: int, after_message_id: int) -> dict | None:
    """Next role='user' message by ID after the given one (used by translation chain)."""
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM messages WHERE session_id = ? AND role = 'user' AND id > ?"
            " ORDER BY id LIMIT 1",
            (session_id, after_message_id),
        ).fetchone()
        return _message(row, payloads.Store(_get_conn())) if row else None


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
        return _message(row, payloads.Store(_get_conn()))


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
    with _lock, _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        payloads.collect(conn)


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
    """Final displayed message, which need not have the greatest stored ID."""
    with _lock:
        rows = _visible_messages(_get_conn(), session_id)
        return rows[-1] if rows else None


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
    with _lock, _get_conn() as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'system'",
            (session_id,),
        )
        payloads.collect(conn)


class GenerationBusy(ValueError):
    pass


def _require_inactive(conn: sqlite3.Connection, session_id: int) -> None:
    if conn.execute(
        "SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')",
        (session_id,),
    ).fetchone():
        raise GenerationBusy("Stop the current response before changing this conversation.")


def _response_session(conn: sqlite3.Connection, session_id: int) -> tuple[dict, dict]:
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if session is None:
        raise ValueError("Chat not found")
    folder = conn.execute("SELECT * FROM folders WHERE id = ?", (session["folder_id"],)).fetchone()
    if folder["type"] == "memo":
        raise ValueError("This folder stores notes")
    return dict(session), dict(folder)


def _touch_session(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))


def _message_api_content(message: dict) -> str | list[dict]:
    if message["role"] == "user":
        return inputs.build_content(message["content"], message["attachments"])
    return message["api_content"] if message["api_content"] is not None else message["content"]


def _history_messages(conn: sqlite3.Connection, rows: list[dict], *,
                      exclude_trace_user_id: int | None = None) -> list[dict]:
    history = []
    store = payloads.Store(conn)
    for row in rows:
        if row["role"] not in ("user", "assistant"):
            continue
        if row["role"] == "assistant":
            variant = conn.execute(
                "SELECT trace_id FROM response_variants WHERE assistant_message_id = ?", (row["id"],)
            ).fetchone()
            if variant and variant["trace_id"]:
                history.extend(api_messages(store.get(variant["trace_id"]), include_failed=True))
        message = {"role": row["role"], "content": _message_api_content(row)}
        if row["role"] == "assistant" and row["reasoning"]:
            message["reasoning"] = row["reasoning"]
        history.append(message)
        if row["role"] == "user" and row["id"] != exclude_trace_user_id:
            pending = conn.execute(
                "SELECT trace_id FROM generations WHERE user_message_id = ? AND assistant_message_id IS NULL",
                (row["id"],),
            ).fetchone()
            if pending and pending["trace_id"]:
                history.extend(api_messages(store.get(pending["trace_id"]), include_failed=True))
    return history


def _enrich_base_reasoning(conn: sqlite3.Connection, user_id: int, request: dict) -> dict:
    """Add recoverable Thinking without replacing any saved request context."""
    if not any(message.get("role") == "assistant" and "reasoning" not in message
               for message in request["messages"]):
        return request
    session_id = conn.execute("SELECT session_id FROM messages WHERE id = ?", (user_id,)).fetchone()[0]
    store = payloads.Store(conn)
    rows = _rows_to_dicts(conn.execute(
        "SELECT m.*, v.user_message_id FROM messages m"
        " LEFT JOIN response_variants v ON v.assistant_message_id = m.id"
        " WHERE m.session_id = ? AND m.role IN ('user', 'assistant') ORDER BY m.id",
        (session_id,),
    ).fetchall())

    def key(message: dict) -> str:
        return json.dumps({name: value for name, value in message.items() if name != "reasoning"},
                          sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    users = {
        row["id"]: key({"role": "user", "content": _message_api_content(_message(row, store))})
        for row in rows if row["role"] == "user" and row["id"] <= user_id
    }
    saved_users = [key(message) for message in request["messages"] if message.get("role") == "user"]
    # Matching the complete user prefix distinguishes repeated identical turns.
    exact_turns = list(users) if saved_users == list(users.values()) else None
    unique_turns = {}
    for turn_id, user_key in users.items():
        unique_turns[user_key] = turn_id if user_key not in unique_turns else None
    candidates = {}

    def remember(turn_id: int | None, message: dict, reasoning: str) -> None:
        if turn_id in users:
            candidates.setdefault((turn_id, key(message)), set()).add(reasoning or "")

    previous_user = None
    for row in rows:
        if row["role"] == "user":
            previous_user = row["id"]
        else:
            remember(row["user_message_id"] or previous_user,
                     {"role": "assistant", "content": _message_api_content(_message(row, store))},
                     row["reasoning"])
    for row in conn.execute(
        "SELECT user_message_id, trace_id FROM generations WHERE session_id = ?"
        " UNION SELECT v.user_message_id, v.trace_id FROM response_variants v"
        " JOIN messages m ON m.id = v.assistant_message_id WHERE m.session_id = ?",
        (session_id, session_id),
    ):
        if row["user_message_id"] not in users or not row["trace_id"]:
            continue
        trace = store.get(row["trace_id"], exclude_keys=frozenset({"request"}))
        for message in api_messages(trace, include_failed=True):
            if message["role"] == "assistant":
                remember(row["user_message_id"], message, message.get("reasoning", ""))
    messages = []
    turn = None
    turn_index = 0
    for message in request["messages"]:
        if message.get("role") == "user":
            turn = exact_turns[turn_index] if exact_turns is not None else unique_turns.get(key(message))
            turn_index += 1
        elif message.get("role") == "assistant" and "reasoning" not in message:
            matches = candidates.get((turn, key(message)), set())
            # Old snapshots have no row IDs. Conflicting identical outputs cannot
            # safely be attributed to a particular historical response version.
            if len(matches) == 1 and (reasoning := next(iter(matches))):
                message = {**message, "reasoning": reasoning}
        messages.append(message)
    return {**request, "messages": messages}


def _set_permission(conn: sqlite3.Connection, user_id: int, web_search: bool | None) -> None:
    row = conn.execute("SELECT * FROM generations WHERE user_message_id = ?", (user_id,)).fetchone()
    store = payloads.Store(conn)
    request = store.get(row["request_id"])
    if web_search is None:
        web_search = request.get("search_permission", bool(request.get("tools")))
    base = _enrich_base_reasoning(conn, user_id, store.get(row["base_request_id"]))
    request = round_request(base, store.get(row["trace_id"]), web_search)
    conn.execute(
        "UPDATE generations SET request_id = ?, base_request_id = ? WHERE user_message_id = ?",
        (store.put(request), store.put(base), user_id),
    )


def _new_generation(conn: sqlite3.Connection, session: dict, folder: dict,
                    user: dict, rows: list[dict], assistant_id: int | None = None) -> None:
    """Legacy rows have no exact saved request; reconstruct once and retain it."""
    prompt = folder["system_prompt"]
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    conn.execute("INSERT OR IGNORE INTO system_prompts VALUES (?, ?)", (prompt_hash, prompt))
    prefix = []
    for row in rows:
        if folder["type"] == "chat" or row["id"] == user["id"]:
            prefix.append(row)
        if row["id"] == user["id"]:
            break
    messages = ([{"role": "system", "content": prompt}] if prompt else []) + _history_messages(conn, prefix)
    store = payloads.Store(conn)
    request_id = store.put({
        "model": session["model"],
        "params": {k: v for k, v in json.loads(session["params_json"]).items() if k in DEFAULT_PARAMS},
        "messages": messages, "tools": [], "system_prompt_hash": prompt_hash,
    })
    conn.execute(
        "INSERT INTO generations (user_message_id, session_id, request_id, base_request_id, trace_id,"
        " batch_id, status, assistant_message_id) VALUES (?, ?, ?, ?, ?, ?, 'idle', ?)",
        (user["id"], session["id"], request_id, request_id, store.put([]), user["id"], assistant_id),
    )


def _queue_generation(conn: sqlite3.Connection, session_id: int, user_id: int) -> dict:
    conn.execute(
        "UPDATE generations SET status = 'queued', content = '', reasoning = '',"
        " error = NULL, attempt = attempt + 1 WHERE user_message_id = ? AND session_id = ?",
        (user_id, session_id),
    )
    _touch_session(conn, session_id)
    return _generation(conn.execute(
        "SELECT * FROM generations WHERE user_message_id = ?", (user_id,)
    ).fetchone(), payloads.Store(conn))


def request_response(session_id: int, user_message_id: int, *, web_search: bool | None = None) -> dict:
    """Queue the final unanswered input without adding another message."""
    with _lock, _get_conn() as conn:
        session, folder = _response_session(conn, session_id)
        _require_inactive(conn, session_id)
        rows = _visible_messages(conn, session_id)
        if not rows or rows[-1]["id"] != user_message_id or rows[-1]["role"] != "user":
            raise ValueError("Request a response only for the final unanswered user message. Fork first.")
        if conn.execute(
            "SELECT 1 FROM generations WHERE user_message_id = ? AND session_id = ?",
            (user_message_id, session_id),
        ).fetchone() is None:
            _new_generation(conn, session, folder, rows[-1], rows)
        if web_search and folder["type"] != "chat":
            raise ValueError("Web search is only available in Chat folders.")
        _set_permission(conn, user_message_id, web_search)
        return _queue_generation(conn, session_id, user_message_id)


def regenerate_response(session_id: int, assistant_message_id: int, *, web_search: bool | None = None) -> dict:
    """Keep the selected answer intact while a replacement is generated."""
    with _lock, _get_conn() as conn:
        session, folder = _response_session(conn, session_id)
        _require_inactive(conn, session_id)
        rows = _visible_messages(conn, session_id)
        if not rows or rows[-1]["id"] != assistant_message_id or rows[-1]["role"] != "assistant":
            raise ValueError("Only the final selected response can be regenerated. Fork first.")
        variant = conn.execute(
            "SELECT user_message_id FROM response_variants WHERE assistant_message_id = ?",
            (assistant_message_id,),
        ).fetchone()
        if variant is None:
            if rows[-1].get("provenance_id"):
                raise ValueError("This standalone response has no linked user message to regenerate.")
            user = next((row for row in reversed(rows[:-1]) if row["role"] == "user"), None)
            if user is None:
                raise ValueError("This response has no preceding user message.")
            existing = conn.execute(
                "SELECT * FROM generations WHERE user_message_id = ?", (user["id"],)
            ).fetchone()
            if existing is None:
                _new_generation(conn, session, folder, user, rows, assistant_message_id)
            else:
                conn.execute(
                    "UPDATE generations SET assistant_message_id = ? WHERE user_message_id = ?",
                    (assistant_message_id, user["id"]),
                )
            conn.execute(
                "INSERT INTO response_variants (assistant_message_id, user_message_id, request_id)"
                " VALUES (?, ?, NULL)", (assistant_message_id, user["id"]),
            )
            user_id = user["id"]
        else:
            user_id = variant["user_message_id"]
            _select_variant(conn, session_id, user_id, assistant_message_id)
        if web_search and folder["type"] != "chat":
            raise ValueError("Web search is only available in Chat folders.")
        store = payloads.Store(conn)
        current = conn.execute("SELECT * FROM generations WHERE user_message_id = ?", (user_id,)).fetchone()
        base = _enrich_base_reasoning(
            conn, user_id, store.get(current["base_request_id"] or current["request_id"])
        )
        if web_search is None:
            request = store.get(current["request_id"])
            base["tools"] = allowed_tools(request.get("search_permission", bool(request.get("tools"))))
        else:
            base["tools"] = allowed_tools(web_search)
        base_id = store.put(base)
        conn.execute(
            "UPDATE generations SET request_id = ?, base_request_id = ?, trace_id = ? WHERE user_message_id = ?",
            (base_id, base_id, store.put([]), user_id),
        )
        return _queue_generation(conn, session_id, user_id)


def _select_variant(conn: sqlite3.Connection, session_id: int,
                    user_id: int, assistant_id: int) -> None:
    variant = conn.execute(
        "SELECT v.request_id, v.base_request_id, v.trace_id, m.content, m.reasoning FROM response_variants v"
        " JOIN messages m ON m.id = v.assistant_message_id"
        " WHERE v.user_message_id = ? AND v.assistant_message_id = ? AND m.session_id = ?",
        (user_id, assistant_id, session_id),
    ).fetchone()
    if variant is None:
        raise ValueError("Response version not found")
    if variant["request_id"] is not None:
        conn.execute(
            "UPDATE generations SET request_id = ?, base_request_id = ?, trace_id = ?"
            " WHERE user_message_id = ? AND session_id = ?",
            (variant["request_id"], variant["base_request_id"], variant["trace_id"], user_id, session_id),
        )
    else:
        conn.execute(
            "UPDATE generations SET request_id = COALESCE(?, base_request_id),"
            " base_request_id = COALESCE(?, base_request_id), trace_id = ?"
            " WHERE user_message_id = ? AND session_id = ?",
            (variant["base_request_id"], variant["base_request_id"],
             variant["trace_id"] or payloads.store(conn, []), user_id, session_id),
        )
    conn.execute(
        "UPDATE generations SET assistant_message_id = ?, status = 'completed',"
        " content = ?, reasoning = ?, error = NULL WHERE user_message_id = ? AND session_id = ?",
        (assistant_id, variant["content"], variant["reasoning"] or "", user_id, session_id),
    )
    payloads.collect(conn)


def _is_final_turn(conn: sqlite3.Connection, session_id: int, user_id: int) -> bool:
    last = conn.execute(
        "SELECT m.id, m.role FROM messages m"
        " LEFT JOIN response_variants v ON v.assistant_message_id = m.id"
        " LEFT JOIN generations g ON g.user_message_id = v.user_message_id"
        " WHERE m.session_id = ? AND"
        " (v.assistant_message_id IS NULL OR g.assistant_message_id = m.id)"
        " ORDER BY COALESCE(v.user_message_id, m.id) DESC,"
        " CASE WHEN v.assistant_message_id IS NULL THEN 0 ELSE 1 END DESC, m.id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if last is None:
        return False
    if last["id"] == user_id and last["role"] == "user":
        return True
    return conn.execute(
        "SELECT 1 FROM response_variants WHERE assistant_message_id = ? AND user_message_id = ?",
        (last["id"], user_id),
    ).fetchone() is not None


def select_response_variant(session_id: int, user_message_id: int, assistant_message_id: int) -> None:
    with _lock, _get_conn() as conn:
        _response_session(conn, session_id)
        _require_inactive(conn, session_id)
        user = conn.execute(
            "SELECT 1 FROM messages WHERE id = ? AND session_id = ? AND role = 'user'",
            (user_message_id, session_id),
        ).fetchone()
        if user is None:
            raise ValueError("User message not found")
        if not _is_final_turn(conn, session_id, user_message_id):
            raise ValueError("Fork first to change a response that has follow-up messages.")
        _select_variant(conn, session_id, user_message_id, assistant_message_id)
        _touch_session(conn, session_id)


def get_message_generation(session_id: int, message_id: int) -> dict | None:
    """Assistant prompts belong to immutable versions, never the mutable job."""
    with _lock:
        conn = _get_conn()
        message = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND session_id = ?", (message_id, session_id)
        ).fetchone()
        if message is None:
            return None
        if message["role"] == "user":
            row = conn.execute(
                "SELECT * FROM generations WHERE session_id = ? AND user_message_id = ?",
                (session_id, message_id),
            ).fetchone()
            return _generation(row, payloads.Store(conn)) if row else None
        if message["role"] == "assistant":
            row = conn.execute(
                "SELECT g.*, v.request_id AS variant_request_id,"
                " v.base_request_id AS variant_base_request_id, v.trace_id AS variant_trace_id,"
                " m.content AS answer_content, m.reasoning AS answer_reasoning"
                " FROM response_variants v JOIN generations g ON g.user_message_id = v.user_message_id"
                " JOIN messages m ON m.id = v.assistant_message_id"
                " WHERE v.assistant_message_id = ? AND g.session_id = ?", (message_id, session_id),
            ).fetchone()
            if row is None:
                store = payloads.Store(conn)
                provenance = store.get(message["provenance_id"]) if message["provenance_id"] else None
                request = provenance.get("source_request") if provenance else None
                if request is None:
                    return None
                return {
                    **request, "request": request, "base_request": None,
                    "session_id": session_id, "user_message_id": None,
                    "assistant_message_id": message_id, "status": "completed", "attempt": 0,
                    "content": message["content"], "reasoning": message["reasoning"] or "",
                    "error": None, "trace": provenance.get("source_trace", []),
                    "provenance": provenance, "request_id": None, "base_request_id": None,
                    "trace_id": None, "batch_id": None,
                }
            if row["variant_request_id"] is None:
                return None
            result = dict(row)
            result.update(request_id=result.pop("variant_request_id"),
                          base_request_id=result.pop("variant_base_request_id"),
                          trace_id=result.pop("variant_trace_id"),
                          assistant_message_id=message_id, status="completed", error=None,
                          content=result.pop("answer_content"), reasoning=result.pop("answer_reasoning") or "")
            return _generation(result, payloads.Store(conn))
        return None


def response_variant_state(session_id: int, user_message_id: int) -> dict:
    """Read navigation metadata without decoding potentially large requests."""
    with _lock:
        conn = _get_conn()
        ids = [row[0] for row in conn.execute(
            "SELECT v.assistant_message_id FROM response_variants v"
            " JOIN messages u ON u.id = v.user_message_id"
            " WHERE u.session_id = ? AND v.user_message_id = ? ORDER BY v.assistant_message_id",
            (session_id, user_message_id),
        )]
        row = conn.execute(
            "SELECT assistant_message_id FROM generations WHERE session_id = ? AND user_message_id = ?",
            (session_id, user_message_id),
        ).fetchone()
        selected = row["assistant_message_id"] if row else None
        active = conn.execute(
            "SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')",
            (session_id,),
        ).fetchone()
        return {
            "ids": ids, "selected_id": selected,
            "index": ids.index(selected) + 1 if selected in ids else 0, "count": len(ids),
            "can_switch": not active and _is_final_turn(conn, session_id, user_message_id),
        }


def _event_snapshot(conn: sqlite3.Connection, session_id: int, user_id: int,
                    event_index: int, trace_id: str) -> tuple[dict, list[dict]]:
    row = conn.execute(
        "SELECT * FROM generations WHERE session_id = ? AND user_message_id = ?",
        (session_id, user_id),
    ).fetchone()
    if row is None or not trace_id or type(event_index) is not int or event_index < 0:
        raise ValueError("Search event not found.")
    job = dict(row)
    variant = conn.execute("SELECT * FROM response_variants WHERE assistant_message_id = ?",
                           (job["assistant_message_id"],)).fetchone()
    if variant and variant["trace_id"] == trace_id:
        for key in ("request_id", "base_request_id", "trace_id"):
            job[key] = variant[key]
    elif job["trace_id"] != trace_id or (variant and job["status"] == "completed"):
        raise ValueError("This search history changed. Reload before choosing an event.")
    trace = payloads.load(conn, trace_id)
    if event_index >= len(trace) or trace[event_index]["status"] not in ("completed", "rejected"):
        raise ValueError("Only completed or rejected search events can be selected.")
    return job, trace


def get_event_generation(session_id: int, user_message_id: int,
                         event_index: int, trace_id: str) -> dict | None:
    with _lock:
        conn = _get_conn()
        job, trace = _event_snapshot(conn, session_id, user_message_id, event_index, trace_id)
        event = trace[event_index]
        if event["kind"] != "assistant":
            return None
        store = payloads.Store(conn)
        result = _generation(job, store)
        result.update(event["request"])
        result.update(request=event["request"], content=event["message"].get("content") or "",
                      reasoning=event.get("reasoning", ""), status="completed", error=None,
                      event_index=event_index)
        result["request_id"] = store.child(store.child(trace_id, event_index), "request")
        return result


class EditConflict(ValueError):
    """The source changed after its editor snapshot was opened."""


def _edit_snapshot(conn: sqlite3.Connection, session_id: int) -> dict:
    source = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if source is None:
        raise ValueError("Chat not found")
    _require_inactive(conn, session_id)
    session = dict(source)
    folder = dict(conn.execute("SELECT * FROM folders WHERE id = ?", (source["folder_id"],)).fetchone())
    rows = _visible_messages(conn, session_id, decode=False)
    jobs = _rows_to_dicts(conn.execute(
        "SELECT * FROM generations WHERE session_id = ? ORDER BY user_message_id", (session_id,)
    ).fetchall())
    variants = _rows_to_dicts(conn.execute(
        "SELECT v.* FROM response_variants v JOIN messages m ON m.id = v.assistant_message_id"
        " WHERE m.session_id = ? ORDER BY v.assistant_message_id", (session_id,)
    ).fetchall())
    revision = hashlib.sha256(json.dumps(
        [session, folder, rows, jobs, variants], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    by_user = {job["user_message_id"]: job for job in jobs}
    by_answer = {variant["assistant_message_id"]: variant for variant in variants}
    store = payloads.Store(conn)
    messages, groups = [], []
    position = 0
    for row in rows:
        if row["role"] == "system":
            continue
        attachments = []
        roots = store.list_children(row["attachments_id"]) if row["attachments_id"] else ()
        for index, root in enumerate(roots):
            descriptor = {"index": index}
            for key in ("name", "media_type", "size", "page_count", "width", "height", "sha256"):
                child = store.child(root, key)
                if child is not None:
                    descriptor[key] = store.get(child)
            attachments.append(descriptor)
        provenance = None
        if row.get("provenance_id"):
            provenance = {}
            for key in ("kind", "source_session_id", "source_message_id"):
                child = store.child(row["provenance_id"], key)
                if child is not None:
                    provenance[key] = store.get(child)
        variant = by_answer.get(row["id"])
        messages.append({
            "id": row["id"], "role": row["role"], "content": row["content"],
            "reasoning": (row["reasoning"] or "") if row["role"] == "assistant" else "",
            "position": position, "user_message_id": variant["user_message_id"] if variant else None,
            "provenance": provenance, "attachments": attachments,
        })
        position += 1
        if row["role"] != "user" or row["id"] not in by_user:
            continue
        job = by_user[row["id"]]
        selected = by_answer.get(job["assistant_message_id"])
        trace_id = selected["trace_id"] if selected else job["trace_id"]
        if not trace_id:
            continue
        trace = store.get(trace_id, exclude_keys=frozenset({"request"}))
        all_calls = [call.get("id") for event in trace if event.get("kind") == "assistant"
                     for call in (event.get("message") or {}).get("tool_calls", [])]
        for start, event in enumerate(trace):
            if event.get("kind") != "assistant":
                continue
            calls = (event.get("message") or {}).get("tool_calls", [])
            end = next((index for index in range(start + 1, len(trace))
                        if trace[index].get("kind") == "assistant"), len(trace))
            indexes = list(range(start, end))
            call_ids = [call.get("id") for call in calls]
            results = trace[start + 1:end]
            complete = bool(calls) and event.get("status") == "completed"
            complete = complete and all(isinstance(call_id, str) and call_id
                                        and all_calls.count(call_id) == 1 for call_id in call_ids)
            complete = complete and len(results) == len(call_ids)
            for call_id in call_ids:
                matches = [result for result in results if result.get("call_id") == call_id]
                complete = complete and len(matches) == 1
                if len(matches) == 1:
                    result = matches[0]
                    message = result.get("message") or {}
                    complete = complete and (
                        result.get("kind") == "tool" and result.get("status") in ("completed", "rejected", "failed")
                        and message.get("role") == "tool" and message.get("tool_call_id") == call_id
                        and isinstance(message.get("content"), str)
                    )
            groups.append({
                "key": f"{row['id']}:{start}", "user_message_id": row["id"],
                "trace_id": trace_id, "position": position, "event_indexes": indexes,
                "complete": bool(complete),
                "reasoning": event.get("reasoning") or "",
                "events": [{**trace[index], "event_index": index} for index in indexes],
            })
            position += 1
    return {"revision": revision, "session": session, "folder": folder,
            "messages": messages, "tool_groups": groups}


def edit_snapshot(session_id: int) -> dict:
    """Describe selected messages without transporting image or request payloads."""
    with _lock:
        return _edit_snapshot(_get_conn(), session_id)


def _edited_provenance(store: payloads.Store, source: dict, session_id: int,
                       kind: str, snapshot: dict | None, edited_fields: list[str]) -> str:
    previous = store.get(source["provenance_id"]) if source.get("provenance_id") else {}
    request_id = snapshot.get("request_id") if snapshot else None
    trace_id = snapshot.get("trace_id") if snapshot else None
    provenance = {
        "kind": kind, "source_session_id": session_id, "source_message_id": source["id"],
        "source_content": source["content"],
        "before_reasoning": source["reasoning"] or "",
        "edited_fields": edited_fields,
        "source_request": previous.get("source_request") if previous else
                          (store.get(request_id) if request_id else None),
        "source_trace": previous.get("source_trace", []) if previous else
                        (store.get(trace_id) if trace_id else []),
        "source_model": previous.get("source_model") if previous else source["model"],
        "source_reasoning": previous.get("source_reasoning") if previous else source["reasoning"],
        "source_usage": previous.get("source_usage") if previous else {
            key: source[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    return store.put(provenance)


def save_edited_fork(session_id: int, draft: dict, *, send: bool = False,
                     web_search: bool = False) -> dict:
    """Validate the entire draft, then atomically clone it and optionally queue one input."""
    if type(send) is not bool or type(web_search) is not bool:
        raise ValueError("Send and web search permissions must be booleans.")
    with _lock, _get_conn() as conn:
        # The reservation precedes every read so another connection cannot change
        # the source between revision validation and the atomic fork commit.
        conn.execute("BEGIN IMMEDIATE")
        snapshot = _edit_snapshot(conn, session_id)
        if not isinstance(draft, dict) or not isinstance(draft.get("revision"), str):
            raise ValueError("An editor revision is required.")
        if draft["revision"] != snapshot["revision"]:
            raise EditConflict("This conversation changed. Reopen the editor before saving; your draft was not saved.")
        edits, choices = draft.get("messages"), draft.get("tools")
        if not isinstance(edits, list) or not isinstance(choices, list):
            raise ValueError("The draft must contain every message and tool group.")
        expected_messages, expected_tools = snapshot["messages"], snapshot["tool_groups"]
        if len(edits) != len(expected_messages) or len(choices) != len(expected_tools):
            raise ValueError("The draft must contain every message and tool group in order.")
        first_change = float("inf")
        included, changed = {}, {}
        for edit, original in zip(edits, expected_messages):
            if (not isinstance(edit, dict) or type(edit.get("id")) is not int
                    or edit["id"] != original["id"] or type(edit.get("include")) is not bool
                    or not isinstance(edit.get("content"), str)
                    or not isinstance(edit.get("reasoning"), str)
                    or not isinstance(edit.get("attachments"), list)):
                raise ValueError("Invalid message edit or message order.")
            if original["role"] != "assistant" and edit["reasoning"]:
                raise ValueError("Only assistant messages may contain Thinking.")
            indexes = edit["attachments"]
            count = len(original["attachments"])
            if (any(type(index) is not int or not 0 <= index < count for index in indexes)
                    or indexes != sorted(set(indexes))):
                raise ValueError("Attachment selections must be unique original indexes in order.")
            if edit["include"]:
                if not edit["content"].strip() and not (
                    original["role"] == "assistant" and edit["reasoning"].strip()
                ) and not (
                    original["role"] == "user" and indexes and snapshot["folder"]["type"] != "memo"
                ):
                    raise ValueError("Included messages require text, assistant Thinking, or a user attachment.")
                included[original["id"]] = edit
            is_changed = (not edit["include"] or edit["content"] != original["content"]
                          or edit["reasoning"] != original["reasoning"]
                          or indexes != list(range(count)))
            changed[original["id"]] = is_changed
            if is_changed:
                first_change = min(first_change, original["position"])
        for choice, group in zip(choices, expected_tools):
            if (not isinstance(choice, dict) or choice.get("key") != group["key"]
                    or type(choice.get("include")) is not bool or type(choice.get("explicit")) is not bool
                    or not isinstance(choice.get("reasoning"), str)):
                raise ValueError("Invalid tool selection or tool group order.")
            if choice["reasoning"] != group["reasoning"]:
                if not group["complete"]:
                    raise ValueError("Incomplete tool group Thinking cannot be edited.")
                first_change = min(first_change, group["position"])
            if choice["include"]:
                if not group["complete"]:
                    raise ValueError("Incomplete tool groups cannot be retained.")
                if group["user_message_id"] not in included:
                    raise ValueError("Tools cannot be retained when their user message is excluded.")
            else:
                first_change = min(first_change, group["position"])
        for choice, group in zip(choices, expected_tools):
            if choice["include"] and group["position"] >= first_change and not choice["explicit"]:
                raise ValueError("Explicitly select tool history to reuse it after a context change.")
        kept = [message for message in expected_messages if message["id"] in included]
        folder = snapshot["folder"]
        if send and (folder["type"] == "memo" or not kept or kept[-1]["role"] != "user"):
            raise ValueError("Save & send requires a final included user message in a model conversation.")
        if web_search and (not send or folder["type"] != "chat"):
            raise ValueError("Web search requires explicit Save & send in a Chat folder.")

        # No inserts or content-addressed allocations occur above this boundary.
        source = snapshot["session"]
        raw = {row["id"]: row for row in _visible_messages(conn, session_id, decode=False)}
        jobs = {row["user_message_id"]: dict(row) for row in conn.execute(
            "SELECT * FROM generations WHERE session_id = ?", (session_id,)
        )}
        variants = {row["assistant_message_id"]: dict(row) for row in conn.execute(
            "SELECT v.* FROM response_variants v JOIN messages m ON m.id = v.assistant_message_id"
            " WHERE m.session_id = ?", (session_id,)
        )}
        cur = conn.execute(
            "INSERT INTO sessions (folder_id, title, auto_title, model, params_json, params_updated_at)"
            " VALUES (?, ?, 0, ?, ?, ?)",
            (source["folder_id"], source["title"], source["model"],
             source["params_json"], source["params_updated_at"]),
        )
        fork_id = cur.lastrowid
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?",
                     (f"{source['title']} · fork #{fork_id}", fork_id))
        fork = dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (fork_id,)).fetchone())
        store = payloads.Store(conn)
        retained = {}
        source_traces = {}
        for choice, group in zip(choices, expected_tools):
            if choice["include"]:
                if group["trace_id"] not in source_traces:
                    source_traces[group["trace_id"]] = store.get(group["trace_id"])
                trace = source_traces[group["trace_id"]]
                events = [{**trace[index], "retained": True} for index in group["event_indexes"]]
                if choice["reasoning"] != group["reasoning"]:
                    event = events[0]
                    previous = event.get("provenance") or {}
                    event["provenance"] = {
                        "kind": "manual", "source_session_id": session_id,
                        "source_user_message_id": group["user_message_id"],
                        "source_event_index": group["event_indexes"][0],
                        "before_reasoning": group["reasoning"],
                        "source_reasoning": previous.get("source_reasoning", group["reasoning"]),
                        "source_usage": previous.get("source_usage", event.get("usage")),
                    }
                    event.update(reasoning=choice["reasoning"], usage=None)
                retained.setdefault(group["user_message_id"], []).extend(events)
        ids, copied_rows = {}, []
        for original in expected_messages:
            old_id = original["id"]
            if old_id not in included:
                continue
            edit, row = included[old_id], raw[old_id]
            copied = {key: value for key, value in row.items() if key != "id"}
            copied.update(session_id=fork_id, content=edit["content"],
                          reasoning=edit["reasoning"] if row["role"] == "assistant" else row["reasoning"])
            attachments = store.get(row["attachments_id"]) if row["attachments_id"] else []
            selected_attachments = [attachments[index] for index in edit["attachments"]]
            if changed[old_id]:
                copied["attachments_id"] = store.put(selected_attachments)
                copied["api_content_id"] = store.put(
                    inputs.build_content(edit["content"], selected_attachments)
                    if row["role"] == "user" else edit["content"]
                )
            orphan = row["role"] == "assistant" and original["user_message_id"] not in included
            kind = ("manual" if row["role"] == "assistant" else "edited") if changed[old_id] else (
                "carried" if row["role"] == "assistant" and
                (original["position"] > first_change or orphan) else None
            )
            if kind:
                historical = variants.get(old_id) if row["role"] == "assistant" else jobs.get(old_id)
                previous = store.get(row["provenance_id"]) if row.get("provenance_id") else {}
                if previous.get("kind") == "manual" and kind == "carried":
                    kind = "manual"
                edited_fields = [field for field in ("content", "reasoning")
                                 if edit[field] != original[field]]
                copied["provenance_id"] = _edited_provenance(
                    store, row, session_id, kind, historical, edited_fields
                )
            if kind == "manual":
                for key in ("model", "reasoning_effort",
                            "prompt_tokens", "completion_tokens", "total_tokens"):
                    copied[key] = None
            columns = ", ".join(copied)
            cur = conn.execute(
                f"INSERT INTO messages ({columns}) VALUES ({', '.join('?' for _ in copied)})",
                tuple(copied.values()),
            )
            ids[old_id] = cur.lastrowid
            copied_rows.append(_message({"id": cur.lastrowid, **copied}, store))

        if folder["type"] != "memo":
            prompt = folder["system_prompt"]
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            conn.execute("INSERT OR IGNORE INTO system_prompts VALUES (?, ?)", (prompt_hash, prompt))
            for original in kept:
                if original["role"] != "user":
                    continue
                old_id, user_id = original["id"], ids[original["id"]]
                prefix = []
                for row in copied_rows:
                    if folder["type"] == "chat" or row["id"] == user_id:
                        prefix.append(row)
                    if row["id"] == user_id:
                        break
                base = {
                    "model": source["model"],
                    "params": {k: v for k, v in json.loads(source["params_json"]).items() if k in DEFAULT_PARAMS},
                    "messages": ([{"role": "system", "content": prompt}] if prompt else [])
                                + _history_messages(conn, prefix, exclude_trace_user_id=user_id),
                    "tools": [], "system_prompt_hash": prompt_hash,
                }
                base_id = store.put(base)
                trace = retained.get(old_id, [])
                trace_id = store.put(trace)
                current_request = store.put({**base, "messages": base["messages"] + api_messages(trace)})
                answer = next((message for message in kept
                               if message["role"] == "assistant" and message["user_message_id"] == old_id), None)
                answer_id = ids[answer["id"]] if answer else None
                historical = variants.get(answer["id"]) if answer else None
                historical_request = historical.get("request_id") if historical else None
                answer_row = next((row for row in copied_rows if row["id"] == answer_id), None)
                conn.execute(
                    "INSERT INTO generations (user_message_id, session_id, request_id, base_request_id,"
                    " trace_id, batch_id, status, content, reasoning, assistant_message_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, fork_id, current_request, base_id, trace_id, user_id,
                     "completed" if answer else "idle", answer_row["content"] if answer else "",
                     (answer_row["reasoning"] or "") if answer else "", answer_id),
                )
                if answer:
                    conn.execute(
                        "INSERT INTO response_variants (assistant_message_id, user_message_id,"
                        " request_id, base_request_id, trace_id) VALUES (?, ?, ?, ?, ?)",
                        (answer_id, user_id, historical_request, base_id, trace_id),
                    )
            if send:
                final_id = ids[kept[-1]["id"]]
                _set_permission(conn, final_id, web_search)
                _queue_generation(conn, fork_id, final_id)
        return dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (fork_id,)).fetchone())


def fork_session(session_id: int, message_id: int, *, event_index: int | None = None,
                 trace_id: str | None = None) -> dict:
    """Clone only a visible prefix and its selected, independent request records."""
    with _lock, _get_conn() as conn:
        source = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if source is None:
            raise ValueError("Chat not found")
        _require_inactive(conn, session_id)
        rows = _visible_messages(conn, session_id, decode=False)
        boundary = next((i for i, row in enumerate(rows)
                         if row["id"] == message_id and row["role"] != "system"), None)
        if boundary is None:
            raise ValueError("Fork boundary must be a visible message in this chat.")
        event_job = None
        event_trace = None
        if event_index is not None:
            if rows[boundary]["role"] != "user":
                raise ValueError("Search event forks must use their user message.")
            event_job, event_trace = _event_snapshot(conn, session_id, message_id, event_index, trace_id)
        elif trace_id is not None:
            raise ValueError("A search history requires an event boundary.")
        prefix = rows[:boundary + 1]
        cur = conn.execute(
            "INSERT INTO sessions (folder_id, title, auto_title, model, params_json, params_updated_at)"
            " VALUES (?, ?, 0, ?, ?, ?)",
            (source["folder_id"], source["title"], source["model"],
             source["params_json"], source["params_updated_at"]),
        )
        fork_id = cur.lastrowid
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?",
                     (f"{source['title']} · fork #{fork_id}", fork_id))
        message_ids = {}
        store = payloads.Store(conn)
        messages_by_id = {message["id"]: message for message in prefix}
        for message in prefix:
            copied = {key: value for key, value in message.items() if key != "id"}
            copied["session_id"] = fork_id
            columns = ", ".join(copied)
            placeholders = ", ".join("?" for _ in copied)
            cur = conn.execute(
                f"INSERT INTO messages ({columns}) VALUES ({placeholders})", tuple(copied.values())
            )
            message_ids[message["id"]] = cur.lastrowid
        jobs = conn.execute(
            "SELECT * FROM generations WHERE session_id = ? ORDER BY user_message_id", (session_id,)
        ).fetchall()
        for row in jobs:
            if row["user_message_id"] not in message_ids:
                continue
            original = dict(row)
            assistant_id = original["assistant_message_id"]
            variant = conn.execute(
                "SELECT request_id, base_request_id, trace_id FROM response_variants WHERE assistant_message_id = ?",
                (assistant_id,),
            ).fetchone() if assistant_id is not None else None
            copied = dict(original)
            if assistant_id in message_ids and variant and variant["request_id"] is not None:
                for key in ("request_id", "base_request_id", "trace_id"):
                    copied[key] = variant[key]
            copied["session_id"] = fork_id
            copied["user_message_id"] = message_ids[original["user_message_id"]]
            copied["assistant_message_id"] = message_ids.get(assistant_id)
            copied["batch_id"] = message_ids.get(original["batch_id"], copied["user_message_id"])
            if copied["assistant_message_id"] is None:
                copied.update(status="idle", content="", reasoning="", error=None)
                snapshot = event_job if original["user_message_id"] == message_id and event_job else original
                base = store.get(snapshot["base_request_id"] or snapshot["request_id"])
                base = (_enrich_base_reasoning(conn, original["user_message_id"], base)
                        if snapshot["base_request_id"] else
                        _clean_base(conn, original["user_message_id"], base))
                base["tools"] = []
                trace = event_trace[:event_index + 1] if event_job and original["user_message_id"] == message_id else []
                copied["base_request_id"] = store.put(base)
                copied["trace_id"] = store.put(trace)
                copied["request_id"] = store.put({**base, "messages": base["messages"] + api_messages(trace)})
            else:
                answer = messages_by_id[assistant_id]
                copied.update(status="completed", content=answer["content"],
                              reasoning=answer["reasoning"] or "", error=None)
                if variant is None or variant["request_id"] is None:
                    base = store.get(original["base_request_id"] or original["request_id"])
                    base = (_enrich_base_reasoning(conn, original["user_message_id"], base)
                            if original["base_request_id"] else
                            _clean_base(conn, original["user_message_id"], base))
                    copied.update(request_id=store.put(base), base_request_id=store.put(base),
                                  trace_id=store.put([]))
            columns = ", ".join(copied)
            placeholders = ", ".join("?" for _ in copied)
            conn.execute(
                f"INSERT INTO generations ({columns}) VALUES ({placeholders})", tuple(copied.values())
            )
            if copied["assistant_message_id"] is not None:
                conn.execute(
                    "INSERT INTO response_variants"
                    " (assistant_message_id, user_message_id, request_id, base_request_id, trace_id)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (copied["assistant_message_id"], copied["user_message_id"],
                     variant["request_id"] if variant else None,
                     variant["base_request_id"] if variant else None,
                     variant["trace_id"] if variant else None),
                )
        if prefix[-1]["role"] == "user":
            terminal_id = message_ids[prefix[-1]["id"]]
            if conn.execute("SELECT 1 FROM generations WHERE user_message_id = ?", (terminal_id,)).fetchone() is None:
                fork = dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (fork_id,)).fetchone())
                folder = dict(conn.execute("SELECT * FROM folders WHERE id = ?", (fork["folder_id"],)).fetchone())
                fork_rows = _visible_messages(conn, fork_id)
                _new_generation(conn, fork, folder, fork_rows[-1], fork_rows)
        return dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (fork_id,)).fetchone())


def enqueue_turns(session_id: int, chunks: list[str], *, attachments: list[dict] | None = None,
                  web_search: bool = False) -> list[dict]:
    """Atomically save input and the exact request, before any network work."""
    attachments = attachments or []
    if (attachments or web_search) and len(chunks) != 1:
        raise ValueError("Attachments and web search require a single Chat message.")
    with _lock, _get_conn() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise ValueError("Chat not found")
        folder = conn.execute("SELECT * FROM folders WHERE id = ?", (session["folder_id"],)).fetchone()
        if folder["type"] == "memo":
            raise ValueError("This folder stores notes")
        if folder["type"] != "chat" and (attachments or web_search):
            raise ValueError("Attachments and web search are only available in Chat folders.")
        if conn.execute(
            "SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')",
            (session_id,),
        ).fetchone():
            raise GenerationBusy("Wait for the current response or stop it before sending another message.")
        for row in conn.execute(
            "SELECT trace_id FROM generations WHERE session_id = ?"
            " AND status = 'idle' AND assistant_message_id IS NULL",
            (session_id,),
        ):
            if pending_calls(payloads.load(conn, row["trace_id"])):
                raise ValueError("Continue the pending response or fork before its tool call.")
        prompt = folder["system_prompt"]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        conn.execute("INSERT OR IGNORE INTO system_prompts VALUES (?, ?)", (prompt_hash, prompt))
        history = _history_messages(conn, _visible_messages(conn, session_id))
        prefix = [{"role": "system", "content": prompt}] if prompt else []
        ids = []
        store = payloads.Store(conn)
        for chunk in chunks:
            api_content = inputs.build_content(chunk, attachments)
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, system_prompt_hash, attachments_id, api_content_id)"
                " VALUES (?, 'user', ?, ?, ?, ?)",
                (session_id, chunk, prompt_hash, store.put(attachments), store.put(api_content)),
            )
            user_id = cur.lastrowid
            messages = prefix + (history if folder["type"] == "chat" else []) + [{"role": "user", "content": api_content}]
            request_id = store.put({
                "model": session["model"],
                "params": {k: v for k, v in json.loads(session["params_json"]).items() if k in DEFAULT_PARAMS},
                "messages": messages, "tools": allowed_tools(web_search), "system_prompt_hash": prompt_hash,
            })
            conn.execute(
                "INSERT INTO generations (user_message_id, session_id, request_id, base_request_id, trace_id, batch_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, session_id, request_id, request_id, store.put([]), ids[0] if ids else user_id),
            )
            ids.append(user_id)
        _refresh_auto_title(conn, session_id)
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))
        return [_message(conn.execute("SELECT * FROM messages WHERE id = ?", (uid,)).fetchone(), store) for uid in ids]


def list_generations(session_id: int) -> list[dict]:
    with _lock:
        conn = _get_conn()
        store = payloads.Store(conn)
        return [_generation(row, store) for row in conn.execute(
            "SELECT * FROM generations WHERE session_id = ? ORDER BY user_message_id", (session_id,)
        ).fetchall()]


def conversation_snapshot(session_id: int) -> tuple[list[dict], list[dict]]:
    """Read replies and generation states together while workers are active."""
    with _lock:
        conn = _get_conn()
        rows = _visible_messages(conn, session_id)
        store = payloads.Store(conn)
        jobs = [_generation(row, store) for row in conn.execute(
            "SELECT * FROM generations WHERE session_id = ? ORDER BY user_message_id", (session_id,)
        ).fetchall()]
        return rows, jobs


def delete_last_message(session_id: int, message_id: int) -> bool:
    with _lock, _get_conn() as conn:
        _require_inactive(conn, session_id)
        rows = _visible_messages(conn, session_id)
        if not rows or rows[-1]["id"] != message_id:
            return False
        selected = conn.execute(
            "SELECT user_message_id FROM response_variants WHERE assistant_message_id = ?",
            (message_id,),
        ).fetchone()
        conn.execute("DELETE FROM messages WHERE id = ? AND session_id = ?", (message_id, session_id))
        if selected is not None:
            user_id = selected["user_message_id"]
            remaining = conn.execute(
                "SELECT assistant_message_id FROM response_variants WHERE user_message_id = ?"
                " ORDER BY ABS(assistant_message_id - ?), assistant_message_id DESC LIMIT 1",
                (user_id, message_id),
            ).fetchone()
            if remaining is not None:
                _select_variant(conn, session_id, user_id, remaining["assistant_message_id"])
            else:
                conn.execute(
                    "UPDATE generations SET status = 'idle', content = '', reasoning = '',"
                    " error = NULL, assistant_message_id = NULL, request_id = base_request_id, trace_id = ?"
                    " WHERE user_message_id = ?",
                    (payloads.store(conn, []), user_id),
                )
        _refresh_auto_title(conn, session_id)
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))
        payloads.collect(conn)
        return True


def get_generation(session_id: int, user_message_id: int) -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM generations WHERE session_id = ? AND user_message_id = ?",
            (session_id, user_message_id),
        ).fetchone()
        return _generation(row, payloads.Store(_get_conn())) if row else None


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
        result = dict(row)
        result["status"] = "running"
        return _generation(result, payloads.Store(conn))


def update_generation(job: dict, content: str, reasoning: str) -> bool:
    with _lock, _get_conn() as conn:
        return conn.execute(
            "UPDATE generations SET content = ?, reasoning = ? WHERE user_message_id = ? AND attempt = ? AND status = 'running'",
            (content, reasoning, job["user_message_id"], job["attempt"]),
        ).rowcount == 1


def _running_job(conn: sqlite3.Connection, job: dict) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM generations WHERE session_id = ? AND user_message_id = ?"
        " AND attempt = ? AND status = 'running'",
        (job["session_id"], job["user_message_id"], job["attempt"]),
    ).fetchone()


def _save_trace(conn: sqlite3.Connection, row: sqlite3.Row, store: payloads.Store,
                trace: list[dict], *, clear_partial: bool = False) -> None:
    request = store.get(row["request_id"])
    base = store.get(row["base_request_id"])
    request = round_request(base, trace, request.get("search_permission", bool(request.get("tools"))))
    conn.execute(
        "UPDATE generations SET request_id = ?, trace_id = ?"
        + (", content = '', reasoning = ''" if clear_partial else "")
        + " WHERE user_message_id = ?",
        (store.put(request), store.put(trace), row["user_message_id"]),
    )
    payloads.collect(conn)


def prepare_model_request(job: dict) -> dict | None:
    """CAS and persist the exact offered budget/tools before contacting the model."""
    with _lock, _get_conn() as conn:
        row = _running_job(conn, job)
        if row is None:
            return None
        store = payloads.Store(conn)
        _save_trace(conn, row, store, store.get(row["trace_id"]))
        return _generation(conn.execute(
            "SELECT * FROM generations WHERE user_message_id = ?", (job["user_message_id"],),
        ).fetchone(), store)


def record_tool_calls(job: dict, message: dict, reasoning: str, usage: dict | None) -> bool:
    with _lock, _get_conn() as conn:
        row = _running_job(conn, job)
        if row is None:
            return False
        store = payloads.Store(conn)
        trace = store.get(row["trace_id"])
        if pending_calls(trace):
            return False
        calls = message.get("tool_calls", [])
        ids = [call["id"] for call in calls]
        previous = {call["id"] for event in trace if event["kind"] == "assistant"
                    for call in event["message"].get("tool_calls", [])}
        if (not calls or any(not isinstance(id_, str) or not id_ for id_ in ids)
                or len(ids) != len(set(ids)) or previous.intersection(ids)):
            raise ValueError("Tool call IDs must be nonempty and unique for this response.")
        trace.append({
            "kind": "assistant", "status": "completed", "origin": "model",
            "message": {**message, "role": "assistant", "content": message.get("content") or ""},
            "reasoning": reasoning or "", "request": store.get(row["request_id"]), "usage": usage,
        })
        _save_trace(conn, row, store, trace, clear_partial=True)
        return True


def start_tool(job: dict, call_id: str) -> bool:
    with _lock, _get_conn() as conn:
        row = _running_job(conn, job)
        if row is None:
            return False
        store = payloads.Store(conn)
        trace = store.get(row["trace_id"])
        call = next((call for call in pending_calls(trace) if call["id"] == call_id), None)
        if call is None:
            return False
        event = {
            "kind": "tool", "status": "running", "origin": "model", "call_id": call_id,
            "name": call["function"]["name"], "arguments": call["function"]["arguments"],
            "started_at": datetime.now(timezone.utc).isoformat(), "message": None,
        }
        existing = next((index for index, old in enumerate(trace)
                         if old["kind"] == "tool" and old["call_id"] == call_id), None)
        if existing is None:
            trace.append(event)
        else:
            event["origin"] = trace[existing]["origin"]
            trace[existing] = event
        _save_trace(conn, row, store, trace)
        return True


def reject_tool(job: dict, call_id: str, error: str, *, error_type: str = "invalid_arguments") -> bool:
    """Resolve an unexecuted call without recording a running search or contacting it."""
    with _lock, _get_conn() as conn:
        row = _running_job(conn, job)
        if row is None:
            return False
        store = payloads.Store(conn)
        trace = store.get(row["trace_id"])
        call = next((call for call in pending_calls(trace) if call["id"] == call_id), None)
        if call is None:
            return False
        origin = next(event["origin"] for event in trace if event["kind"] == "assistant"
                      and any(item["id"] == call_id for item in event["message"]["tool_calls"]))
        result = {"error": error, "error_type": error_type, "executed": False}
        event = {
            "kind": "tool", "status": "rejected", "origin": origin, "call_id": call_id,
            "name": call["function"]["name"], "arguments": call["function"]["arguments"],
            "rejected_at": datetime.now(timezone.utc).isoformat(), "error": error,
            "message": {"role": "tool", "tool_call_id": call_id, "name": call["function"]["name"],
                        "content": json.dumps(result, ensure_ascii=False)},
        }
        existing = next((index for index, old in enumerate(trace)
                         if old["kind"] == "tool" and old["call_id"] == call_id), None)
        if existing is None:
            trace.append(event)
        else:
            trace[existing] = event
        _save_trace(conn, row, store, trace)
        return True


def finish_tool(job: dict, call_id: str, result: dict, *, failed: bool = False) -> bool:
    with _lock, _get_conn() as conn:
        row = _running_job(conn, job)
        if row is None:
            return False
        store = payloads.Store(conn)
        trace = store.get(row["trace_id"])
        event = next((event for event in trace if event["kind"] == "tool"
                      and event["call_id"] == call_id and event["status"] == "running"), None)
        if event is None:
            return False
        event.update(
            status="failed" if failed else "completed",
            message={"role": "tool", "tool_call_id": call_id, "name": event["name"],
                     "content": json.dumps(result, ensure_ascii=False)},
        )
        if failed:
            event["error"] = result.get("error", "Search failed.")
        else:
            event.pop("error", None)
        _save_trace(conn, row, store, trace)
        return True


def finish_generation(job: dict, content: str, reasoning: str, usage: dict | None) -> None:
    usage = usage or {}
    with _lock, _get_conn() as conn:
        row = conn.execute("SELECT * FROM generations WHERE user_message_id = ?", (job["user_message_id"],)).fetchone()
        if row is None or row["status"] != "running" or row["attempt"] != job["attempt"]:
            return
        store = payloads.Store(conn)
        model = store.get(store.child(row["request_id"], "model"))
        params = store.get(store.child(row["request_id"], "params"))
        prompt_hash = store.get(store.child(row["request_id"], "system_prompt_hash"))
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, reasoning, model, reasoning_effort,"
            " prompt_tokens, completion_tokens, total_tokens, system_prompt_hash)"
            " VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["session_id"], content, reasoning or None, model,
             params.get("reasoning_effort"), usage.get("prompt_tokens"),
             usage.get("completion_tokens"), usage.get("total_tokens"), prompt_hash),
        )
        conn.execute(
            "UPDATE generations SET status = 'completed', content = ?, reasoning = ?, error = NULL, assistant_message_id = ?"
            " WHERE user_message_id = ?", (content, reasoning, cur.lastrowid, job["user_message_id"]),
        )
        conn.execute(
            "INSERT INTO response_variants"
            " (assistant_message_id, user_message_id, request_id, base_request_id, trace_id)"
            " VALUES (?, ?, ?, ?, ?)",
            (cur.lastrowid, job["user_message_id"], row["request_id"], row["base_request_id"], row["trace_id"]),
        )
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (job["session_id"],))


def _fail_pending_tools(conn: sqlite3.Connection, row: sqlite3.Row, error: str) -> None:
    store = payloads.Store(conn)
    trace = store.get(row["trace_id"])
    unresolved = pending_calls(trace)
    changed = False
    timestamp = datetime.now(timezone.utc).isoformat()
    content = json.dumps({"error": error}, ensure_ascii=False)
    for call in unresolved:
        event = next((event for event in trace if event["kind"] == "tool"
                      and event["call_id"] == call["id"]), None)
        if event is not None and event["status"] == "failed":
            continue
        if event is None:
            origin = next(event["origin"] for event in trace if event["kind"] == "assistant"
                          and any(item["id"] == call["id"] for item in event["message"]["tool_calls"]))
            event = {"kind": "tool", "origin": origin, "call_id": call["id"],
                     "name": call["function"]["name"], "arguments": call["function"]["arguments"],
                     "started_at": timestamp}
            trace.append(event)
        event.update(
            status="failed", error=error,
            message={"role": "tool", "tool_call_id": call["id"], "name": event["name"], "content": content},
        )
        changed = True
    if changed:
        _save_trace(conn, row, store, trace)


def interrupt_generations(session_id: int, error: str, status: str = "failed", job: dict | None = None) -> None:
    with _lock, _get_conn() as conn:
        if job is not None:
            current = conn.execute("SELECT status, attempt FROM generations WHERE user_message_id = ?", (job["user_message_id"],)).fetchone()
            if current is None or current["status"] != "running" or current["attempt"] != job["attempt"]:
                return
        for row in conn.execute(
            "SELECT * FROM generations WHERE session_id = ? AND status IN ('queued', 'running')",
            (session_id,),
        ):
            _fail_pending_tools(conn, row, error)
        changed = conn.execute(
            "UPDATE generations SET status = ?, error = ? WHERE session_id = ? AND status IN ('queued', 'running')",
            (status, error, session_id),
        ).rowcount
        if changed:
            conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))


def retry_generations(session_id: int, *, web_search: bool | None = None) -> bool:
    with _lock, _get_conn() as conn:
        _, folder = _response_session(conn, session_id)
        if web_search and folder["type"] != "chat":
            raise ValueError("Web search is only available in Chat folders.")
        if conn.execute(
            "SELECT 1 FROM generations WHERE session_id = ? AND status IN ('queued', 'running')", (session_id,)
        ).fetchone():
            raise GenerationBusy("A response is already in progress.")
        # Only retry the latest submission: old failures must never be silently replayed.
        latest = conn.execute("SELECT MAX(id) FROM messages WHERE session_id = ? AND role = 'user'", (session_id,)).fetchone()[0]
        last_job = conn.execute("SELECT status, batch_id FROM generations WHERE user_message_id = ?", (latest,)).fetchone()
        if last_job is None or last_job["status"] not in ("failed", "cancelled"):
            return False
        for row in conn.execute(
            "SELECT user_message_id FROM generations WHERE session_id = ? AND batch_id = ?"
            " AND status IN ('failed', 'cancelled')", (session_id, last_job["batch_id"]),
        ).fetchall():
            _set_permission(conn, row["user_message_id"], web_search)
        conn.execute(
            "UPDATE generations SET status = 'queued', content = '', reasoning = '', error = NULL, attempt = attempt + 1"
            " WHERE session_id = ? AND batch_id = ? AND status IN ('failed', 'cancelled')",
            (session_id, last_job["batch_id"]),
        )
        conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,))
        return True
