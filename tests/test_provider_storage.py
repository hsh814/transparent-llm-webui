"""Offline coverage for provider pairs and immutable request ownership."""

import json
import os
import sqlite3
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import db
import generation
import ollama
import payloads


REGISTRATIONS = [
    {"id": "alpha", "name": "Alpha", "base_url": "https://alpha.example/v1",
     "api_key_env": "ALPHA_TEST_KEY"},
    {"id": "beta", "name": "Beta", "base_url": "https://beta.example/v1",
     "api_key_env": "BETA_TEST_KEY", "default_model": "beta-default"},
    {"id": "ollama", "name": "Ollama", "kind": "ollama", "base_url": "https://ollama.example/v1",
     "api_key_env": "OLLAMA_TEST_KEY", "default_model": "gemma4:31b"},
]


def search_call():
    return {"id": "search-1", "type": "function", "function": {
        "name": "web_search", "arguments": '{"query":"saved query","max_results":2}',
    }}


def draft_for(session_id):
    snapshot = db.edit_snapshot(session_id)
    return {
        "revision": snapshot["revision"],
        "messages": [{"id": message["id"], "include": True, "content": message["content"],
                      "reasoning": message["reasoning"],
                      "attachments": [item["index"] for item in message["attachments"]]}
                     for message in snapshot["messages"]],
        "tools": [{"key": group["key"], "include": group["complete"], "explicit": True,
                   "reasoning": group["reasoning"]} for group in snapshot["tool_groups"]],
    }


class ProviderStorageTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(TemporaryDirectory())
        self.path = Path(directory) / "providers.db"
        config_path = Path(directory) / "providers.json"
        config_path.write_text(json.dumps(REGISTRATIONS), encoding="utf-8")
        self.stack.enter_context(patch.object(db, "DB_PATH", self.path))
        self.stack.enter_context(patch.object(db, "_conn", None))
        self.stack.enter_context(patch.dict(os.environ, {
            "LLM_PROVIDERS_FILE": str(config_path),
            "ALPHA_TEST_KEY": "alpha-secret", "BETA_TEST_KEY": "beta-secret",
            "OLLAMA_TEST_KEY": "ollama-secret",
        }))
        self.stack.callback(lambda: db._conn.close() if db._conn is not None else None)
        db.init_db()
        self.folder = db.create_folder("Pairs", "Be precise.")

    def session(self, provider="alpha", model="shared", folder=None):
        return db.create_session((folder or self.folder)["id"], provider=provider, model=model)

    def completed(self, session):
        db.enqueue_turns(session["id"], ["Question"])
        job = db.claim_generation(session["id"])
        db.finish_generation(job, "Answer", "Thinking", {"total_tokens": 5})
        return db.get_generation(session["id"], job["user_message_id"])

    def test_defaults_and_usage_are_provider_specific(self):
        empty = db.create_session(self.folder["id"])
        self.assertEqual((empty["provider"], empty["model"]), ("alpha", ""))
        self.assertEqual(db.default_model("beta"), "beta-default")
        self.assertEqual(db.default_model("ollama"), "gemma4:31b")
        self.session("alpha", "shared")
        self.session("alpha", "alpha-only")
        self.session("alpha", "alpha-only")
        for _ in range(3):
            self.session("beta", "shared")
        self.assertEqual(db.model_usage("alpha"), {"alpha-only": 2, "shared": 1})
        self.assertEqual(db.model_usage("beta"), {"shared": 3})
        self.assertEqual(db.default_model("alpha"), "alpha-only")
        self.assertEqual(db.default_model("beta"), "shared")
        created = db.create_session(self.folder["id"])
        self.assertEqual((created["provider"], created["model"]), ("alpha", "alpha-only"))

    def test_pair_inheritance_never_crosses_provider_boundary(self):
        source = self.session("beta", "beta-chosen")
        db.update_session(source["id"], params_json='{"temperature":0.2}', params_updated_at="2026-01-01")
        inherited = db.create_session(self.folder["id"])
        self.assertEqual((inherited["provider"], inherited["model"]), ("beta", "beta-chosen"))
        self.assertEqual(inherited["params_json"], '{"temperature":0.2}')
        explicit = db.create_session(self.folder["id"], provider="alpha")
        self.assertEqual((explicit["provider"], explicit["model"]), ("alpha", ""))
        with patch.dict(os.environ, {"BETA_TEST_KEY": "", "OLLAMA_TEST_KEY": ""}):
            fresh = db.create_session(self.folder["id"])
            self.assertEqual((fresh["provider"], fresh["model"]), ("alpha", ""))
            with self.assertRaisesRegex(ValueError, "not configured"):
                db.default_model("beta")
        self.assertEqual(db.get_session(source["id"])["provider"], "beta")

    def test_queued_translation_pair_survives_selection_and_retry(self):
        folder = db.create_folder("Translations", folder_type="translation")
        session = self.session(folder=folder)
        users = db.enqueue_turns(session["id"], ["First", "Second"])
        db.update_session(session["id"], provider="beta", model="other")
        first = db.claim_generation(session["id"])
        self.assertEqual((first["provider"], first["model"]), ("alpha", "shared"))
        db.finish_generation(first, "Translated", "", None)
        db.interrupt_generations(session["id"], "Interrupted")
        self.assertTrue(db.retry_generations(session["id"]))
        second = db.claim_generation(session["id"])
        self.assertEqual(second["user_message_id"], users[1]["id"])
        self.assertEqual((second["provider"], second["model"]), ("alpha", "shared"))
        db.finish_generation(second, "Translated again", "", None)
        for answer in db.visible_messages(session["id"]):
            if answer["role"] == "assistant":
                self.assertEqual((answer["provider"], answer["model"]), ("alpha", "shared"))

    def test_regeneration_and_forks_keep_saved_pair(self):
        session = self.session()
        completed = self.completed(session)
        db.update_session(session["id"], provider="beta", model="other")
        request = db.regenerate_response(session["id"], completed["assistant_message_id"])
        self.assertEqual((request["provider"], request["model"]), ("alpha", "shared"))
        db.interrupt_generations(session["id"], "Stopped", status="cancelled")
        original = db.get_message_generation(session["id"], completed["assistant_message_id"])
        self.assertEqual(original["provider"], "alpha")
        for boundary in (completed["user_message_id"], completed["assistant_message_id"]):
            fork = db.fork_session(session["id"], boundary)
            self.assertEqual((fork["provider"], fork["model"]), ("beta", "other"))
            saved = db.list_generations(fork["id"])[0]
            self.assertEqual((saved["provider"], saved["model"]), ("alpha", "shared"))
            self.assertEqual(saved["base_request"]["provider"], "alpha")
            if boundary == completed["user_message_id"]:
                requested = db.request_response(fork["id"], saved["user_message_id"])
                self.assertEqual(requested["provider"], "alpha")

    def test_edited_forks_clear_attribution_but_keep_original_source_pair(self):
        session = self.session()
        completed = self.completed(session)
        db.update_session(session["id"], provider="beta", model="other")
        draft = draft_for(session["id"])
        draft["messages"][-1]["content"] = "Manual answer"
        fork = db.save_edited_fork(session["id"], draft)
        answer = db.visible_messages(fork["id"])[-1]
        self.assertIsNone(answer["provider"])
        self.assertIsNone(answer["model"])
        self.assertEqual(answer["provenance"]["source_provider"], "alpha")
        self.assertEqual(answer["provenance"]["source_model"], "shared")
        self.assertEqual(answer["provenance"]["source_request"]["provider"], "alpha")
        draft = draft_for(fork["id"])
        draft["messages"][-1]["reasoning"] = "Manual thinking"
        repeated = db.save_edited_fork(fork["id"], draft)
        self.assertEqual(db.visible_messages(repeated["id"])[-1]["provenance"]["source_provider"], "alpha")
        draft = draft_for(session["id"])
        draft["messages"][-1]["include"] = False
        draft["messages"][0]["content"] = "Edited question"
        sent = db.save_edited_fork(session["id"], draft, send=True)
        job = db.list_generations(sent["id"])[0]
        self.assertEqual((job["provider"], job["model"], job["status"]), ("beta", "other", "queued"))
        self.assertEqual(job["messages"][-1]["content"], "Edited question")
        self.assertEqual(db.get_message_generation(session["id"], completed["assistant_message_id"])["provider"], "alpha")

    def test_legacy_answer_without_snapshot_uses_current_pair(self):
        session = self.session("ollama", "original")
        db.add_message(session["id"], "user", "Old question")
        answer = db.add_message(session["id"], "assistant", "Old answer", model="original", provider="ollama")
        db.update_session(session["id"], provider="beta", model="other")
        requested = db.regenerate_response(session["id"], answer["id"])
        self.assertEqual((requested["provider"], requested["model"]), ("beta", "other"))

    def test_empty_model_and_unsupported_new_web_request_are_atomic(self):
        empty = db.create_session(self.folder["id"])
        with self.assertRaisesRegex(ValueError, "Select a model"):
            db.enqueue_turns(empty["id"], ["Question"])
        user = db.add_message(empty["id"], "user", "Draft question")
        with self.assertRaisesRegex(ValueError, "Select a model"):
            db.request_response(empty["id"], user["id"])
        before = len(db.list_sessions(self.folder["id"]))
        with self.assertRaisesRegex(ValueError, "Select a model"):
            db.save_edited_fork(empty["id"], draft_for(empty["id"]), send=True)
        self.assertEqual(len(db.list_sessions(self.folder["id"])), before)
        self.assertEqual(db.list_generations(empty["id"]), [])
        session = self.session()
        with self.assertRaisesRegex(ValueError, "Ollama"):
            db.enqueue_turns(session["id"], ["Question"], web_search=True)
        self.assertEqual(db.list_messages(session["id"]), [])

    def test_removed_provider_never_retargets_queued_or_retry_work(self):
        session = self.session()
        user = db.enqueue_turns(session["id"], ["Question"])[0]
        db.update_session(session["id"], provider="beta", model="other")
        with patch.dict(os.environ, {"ALPHA_TEST_KEY": ""}), \
                patch.object(ollama, "chat_stream") as stream:
            generation._generate(db.claim_generation(session["id"]))
            stream.assert_not_called()
            saved = db.get_generation(session["id"], user["id"])
            self.assertEqual(saved["provider"], "alpha")
            self.assertEqual(saved["status"], "failed")
            self.assertIn("not configured", saved["error"])
            with self.assertRaisesRegex(ValueError, "not configured"):
                db.retry_generations(session["id"])
        self.assertTrue(db.retry_generations(session["id"]))
        self.assertEqual(db.get_generation(session["id"], user["id"])["provider"], "alpha")

    def test_worker_routes_models_and_tools_using_saved_provider(self):
        session = self.session("ollama")
        db.enqueue_turns(session["id"], ["Question"], web_search=True)
        db.update_session(session["id"], provider="beta", model="other")
        routed = []

        def stream(model, messages, params, *, tools=None, provider):
            routed.append((provider, model))
            if messages[-1].get("name") == "web_fetch":
                yield {"type": "content", "text": "Answer"}
                yield {"type": "done", "finish_reason": "stop"}
            elif messages[-1]["role"] == "tool":
                yield {"type": "done", "finish_reason": "tool_calls", "tool_calls": [{
                    "id": "fetch-1", "type": "function", "function": {
                        "name": "web_fetch", "arguments": '{"url":"https://example.org/page"}',
                    },
                }]}
            else:
                yield {"type": "done", "finish_reason": "tool_calls", "tool_calls": [search_call()]}

        with patch.object(ollama, "chat_stream", side_effect=stream), \
                patch.object(ollama, "web_search", return_value=[]) as search, \
                patch.object(ollama, "web_fetch", return_value={"content": "Saved page"}) as fetch:
            generation._generate(db.claim_generation(session["id"]))
        self.assertEqual(routed, [("ollama", "shared")] * 3)
        search.assert_called_once_with("saved query", max_results=2, provider="ollama")
        fetch.assert_called_once_with("https://example.org/page", provider="ollama")
        job = db.list_generations(session["id"])[0]
        self.assertEqual(job["trace"][0]["request"]["provider"], "ollama")
        fork = db.fork_session(session["id"], job["user_message_id"], event_index=0, trace_id=job["trace_id"])
        self.assertEqual(db.list_generations(fork["id"])[0]["provider"], "ollama")

    def test_unsupported_saved_tool_fails_at_execution_without_http(self):
        session = self.session()
        user = db.enqueue_turns(session["id"], ["Question"])[0]
        job = db.claim_generation(session["id"])
        db.record_tool_calls(job, {"role": "assistant", "content": "", "tool_calls": [search_call()]}, "", None)
        with db._lock, db._get_conn() as conn:
            row = conn.execute("SELECT request_id FROM generations WHERE user_message_id = ?", (user["id"],)).fetchone()
            request = payloads.load(conn, row["request_id"])
            request["search_permission"] = True
            conn.execute("UPDATE generations SET request_id = ? WHERE user_message_id = ?",
                         (payloads.store(conn, request), user["id"]))
        with patch.object(ollama.httpx, "Client") as client, patch.object(ollama, "chat_stream") as stream:
            generation._generate(db.get_generation(session["id"], user["id"]))
            client.assert_not_called()
            stream.assert_not_called()
        failed = db.get_generation(session["id"], user["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("not supported", failed["error"])
        self.assertEqual(failed["trace"][-1]["status"], "failed")

    def legacy_database(self):
        db._conn.close()
        db._conn = None
        self.path.unlink()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(payloads.SCHEMA + db.SCHEMA.replace(
                "  provider TEXT NOT NULL DEFAULT 'ollama',\n", "",
            ).replace("  provider TEXT,\n", ""))
            conn.execute("PRAGMA user_version = 1")
            conn.execute("INSERT INTO folders (id, name) VALUES (1, 'Legacy')")
            for sid in (1, 2):
                conn.execute("INSERT INTO sessions (id, folder_id, model) VALUES (?, 1, 'shared')", (sid,))
            store = payloads.Store(conn)
            request = {"model": "shared", "params": {}, "messages": [
                {"role": "user", "content": '{"provider":"user text"}'},
            ], "tools": [], "system_prompt_hash": None}
            trace = [{"kind": "assistant", "status": "completed", "origin": "model",
                      "message": {"role": "assistant", "content": "", "tool_calls": [search_call()]},
                      "request": request, "reasoning": "Edited thinking", "usage": None,
                      "provenance": {"kind": "manual", "source_reasoning": "Original thinking"}}]
            provenance = {"kind": "manual", "source_model": "shared", "source_request": request,
                          "source_trace": trace, "source_content": "Answer"}
            request_id, trace_id = store.put(request), store.put(trace)
            for mid, sid, role, content, model in (
                (1, 1, "user", '{"provider":"user text"}', None),
                (2, 1, "assistant", "Answer", "shared"),
                (3, 1, "assistant", "Manual answer", None),
                (4, 2, "user", '{"provider":"user text"}', None),
                (5, 2, "assistant", "Answer", "shared"),
                (6, 1, "assistant", "Unknown snapshot", "older"),
                (7, 1, "assistant", "Historical snapshot", "older"),
            ):
                conn.execute("INSERT INTO messages (id, session_id, role, content, model, provenance_id)"
                             " VALUES (?, ?, ?, ?, ?, ?)",
                             (mid, sid, role, content, model, store.put(provenance) if mid == 3 else None))
            for user, sid, answer in ((1, 1, 2), (4, 2, 5)):
                conn.execute("INSERT INTO generations (user_message_id, session_id, request_id, base_request_id,"
                             " trace_id, assistant_message_id, status) VALUES (?, ?, ?, ?, ?, ?, 'completed')",
                             (user, sid, request_id, request_id, trace_id, answer))
                conn.execute("INSERT INTO response_variants VALUES (?, ?, ?, ?, ?)",
                             (answer, user, request_id, request_id, trace_id))
            conn.execute("INSERT INTO response_variants VALUES (6, 1, NULL, NULL, NULL)")
            historical_id = store.put({**request, "model": "older"})
            conn.execute("INSERT INTO response_variants VALUES (7, 1, ?, ?, ?)",
                         (historical_id, historical_id, store.put([])))
        return request_id, trace_id

    def test_legacy_migration_rewrites_shared_roots_and_preserves_unknowns(self):
        old_request, old_trace = self.legacy_database()
        db.init_db()
        self.assertEqual(db.get_session(1)["provider"], "ollama")
        self.assertEqual(db.get_session(2)["provider"], "ollama")
        first, second = db.list_generations(1)[0], db.list_generations(2)[0]
        self.assertEqual(first["provider"], "ollama")
        self.assertEqual(first["base_request"]["provider"], "ollama")
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(first["request_id"], first["base_request_id"])
        self.assertNotEqual(first["request_id"], old_request)
        self.assertNotEqual(first["trace_id"], old_trace)
        self.assertEqual(first["messages"][-1]["content"], '{"provider":"user text"}')
        self.assertEqual(first["trace"][0]["request"]["provider"], "ollama")
        self.assertEqual(first["trace"][0]["provenance"]["source_provider"], "ollama")
        messages = {message["id"]: message for message in db.list_messages(1)}
        self.assertIsNone(messages[1]["provider"])
        self.assertEqual(messages[2]["provider"], "ollama")
        self.assertIsNone(messages[3]["provider"])
        self.assertEqual(messages[3]["provenance"]["source_provider"], "ollama")
        self.assertEqual(messages[3]["provenance"]["source_request"]["provider"], "ollama")
        self.assertEqual(messages[3]["provenance"]["source_trace"][0]["request"]["provider"], "ollama")
        self.assertIsNone(db.get_message_generation(1, 6))
        historical = db.get_message_generation(1, 7)
        self.assertEqual((historical["provider"], historical["model"]), ("ollama", "older"))
        with db._lock:
            conn = db._get_conn()
            store = payloads.Store(conn)
            self.assertEqual(store.child(messages[3]["provenance_id"], "source_request"), first["request_id"])
            self.assertEqual(conn.execute("SELECT request_id FROM response_variants WHERE assistant_message_id = 2").fetchone()[0], first["request_id"])
            before = list(conn.iterdump())
        db.init_db()
        with db._lock:
            self.assertEqual(list(db._get_conn().iterdump()), before)
            self.assertEqual(db._get_conn().execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_provider_migration_rolls_back_columns_roots_and_version(self):
        old_request, _ = self.legacy_database()
        put = payloads.Store.put

        def interrupted(store, value):
            if isinstance(value, dict) and value.get("provider") == "ollama" and value.get("model") == "shared":
                raise RuntimeError("Interrupted migration")
            return put(store, value)

        with patch.object(payloads.Store, "put", interrupted), self.assertRaisesRegex(RuntimeError, "Interrupted migration"):
            db.init_db()
        with db._lock:
            conn = db._get_conn()
            self.assertNotIn("provider", {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")})
            self.assertNotIn("provider", {row["name"] for row in conn.execute("PRAGMA table_info(messages)")})
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT request_id FROM generations WHERE user_message_id = 1").fetchone()[0], old_request)
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        db.init_db()
        self.assertEqual(db.list_generations(1)[0]["provider"], "ollama")
