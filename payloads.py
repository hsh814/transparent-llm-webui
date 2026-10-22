"""Content-addressed JSON nodes used by immutable conversation inputs."""

import hashlib
import json
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS payloads (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('scalar', 'list', 'object')),
  body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS payload_edges (
  parent_id TEXT NOT NULL REFERENCES payloads(id) ON DELETE CASCADE,
  child_id TEXT NOT NULL REFERENCES payloads(id) ON DELETE CASCADE,
  PRIMARY KEY (parent_id, child_id)
);
CREATE INDEX IF NOT EXISTS payload_edges_child ON payload_edges(child_id);
"""


class Store:
    """Operation-local caches; no mutable decoded values outlive an operation."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._stored: set[str] = set()
        self._nodes: dict[str, tuple[str, object]] = {}

    def put(self, value) -> str:
        if isinstance(value, dict):
            kind = "object"
            body_value = {key: self.put(child) for key, child in value.items()}
            children = body_value.values()
        elif isinstance(value, list):
            kind = "list"
            body_value = [self.put(child) for child in value]
            children = body_value
        else:
            kind = "scalar"
            body_value = value
            children = ()
        body = json.dumps(body_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        id_ = hashlib.sha256((kind + "\0" + body).encode()).hexdigest()
        if id_ not in self._stored:
            self.conn.execute("INSERT OR IGNORE INTO payloads (id, kind, body) VALUES (?, ?, ?)",
                              (id_, kind, body))
            self.conn.executemany(
                "INSERT OR IGNORE INTO payload_edges (parent_id, child_id) VALUES (?, ?)",
                ((id_, child) for child in set(children)),
            )
            self._stored.add(id_)
        return id_

    def _node(self, id_: str) -> tuple[str, object]:
        if id_ not in self._nodes:
            row = self.conn.execute("SELECT kind, body FROM payloads WHERE id = ?", (id_,)).fetchone()
            if row is None:
                raise ValueError(f"Missing immutable payload: {id_}")
            self._nodes[id_] = (row["kind"], json.loads(row["body"]))
        return self._nodes[id_]

    def child(self, id_: str, key: str | int) -> str | None:
        """Read an edge without decoding any potentially large descendants."""
        kind, body = self._node(id_)
        if kind == "object":
            return body.get(key)
        if kind == "list":
            return body[key] if body else None
        raise ValueError("Scalar payloads have no children")

    def get(self, id_: str):
        kind, body = self._node(id_)
        if kind == "object":
            return {key: self.get(child) for key, child in body.items()}
        if kind == "list":
            return [self.get(child) for child in body]
        return body


def store(conn: sqlite3.Connection, value) -> str:
    return Store(conn).put(value)


def load(conn: sqlite3.Connection, id_: str):
    return Store(conn).get(id_)


def collect(conn: sqlite3.Connection) -> None:
    """Reclaim only nodes unreachable from live roots in the caller's transaction."""
    conn.execute("""
        WITH RECURSIVE live(id) AS (
          SELECT attachments_id FROM messages WHERE attachments_id IS NOT NULL
          UNION SELECT api_content_id FROM messages WHERE api_content_id IS NOT NULL
          UNION SELECT request_id FROM generations
          UNION SELECT base_request_id FROM generations WHERE base_request_id IS NOT NULL
          UNION SELECT trace_id FROM generations WHERE trace_id IS NOT NULL
          UNION SELECT request_id FROM response_variants WHERE request_id IS NOT NULL
          UNION SELECT base_request_id FROM response_variants WHERE base_request_id IS NOT NULL
          UNION SELECT trace_id FROM response_variants WHERE trace_id IS NOT NULL
          UNION SELECT e.child_id FROM payload_edges e JOIN live ON e.parent_id = live.id
        )
        DELETE FROM payloads WHERE id NOT IN (SELECT id FROM live)
    """)
    # Startup migration intentionally has FK enforcement disabled until its
    # atomic table replacement is complete, so do not rely solely on cascades.
    conn.execute(
        "DELETE FROM payload_edges WHERE parent_id NOT IN (SELECT id FROM payloads)"
        " OR child_id NOT IN (SELECT id FROM payloads)"
    )
