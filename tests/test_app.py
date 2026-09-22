"""Offline regressions. Run: uv run python -m unittest discover -s tests -v."""

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from html.parser import HTMLParser
from unittest.mock import patch

os.environ.setdefault("OLLAMA_API_KEY", "test-key")

import httpx
from fastapi.testclient import TestClient

import db
import generation
import ollama
import templating
from app import app
from routers.chat import _split_translation, _sse

START_WORKER = generation.start


class ElementIds(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])


class AppTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(TemporaryDirectory())
        self.stack.enter_context(patch.object(db, "DB_PATH", Path(directory) / "test.db"))
        self.stack.enter_context(patch.object(db, "_conn", None))
        self.stack.enter_context(patch.object(ollama, "list_models", return_value=["test-model", "other-model"]))
        self.stack.enter_context(patch.object(ollama, "model_context_length", return_value=32768))
        self.start = self.stack.enter_context(patch.object(generation, "start"))
        self.client = self.stack.enter_context(TestClient(app))
        self.stack.callback(lambda: db._conn.close())
        self.folder = db.create_folder("Personal", "Be exact.")
        self.session = db.create_session(self.folder["id"], model="test-model")
        self.sid = self.session["id"]

    def send(self, content="Hello"):
        return self.client.post(f"/sessions/{self.sid}/send", data={"content": content})

    def finish(self, text="Answer\n\n    code", reasoning="", usage=None):
        job = db.claim_generation(self.sid)
        db.finish_generation(job, text, reasoning, usage)
        return db.get_generation(self.sid, job["user_message_id"])

    def test_first_visit_and_direct_links_render_full_pages(self):
        for path in ("/", f"/sessions/{self.sid}", f"/?session={self.sid}"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn('<html lang="en">', response.text)
            self.assertIn('id="active-session"', response.text)
        response = self.client.get(f"/sessions/{self.sid}", headers={"HX-Request": "true"})
        self.assertNotIn('<html', response.text)

    def test_send_snapshots_prompt_model_parameters_and_history(self):
        self.assertEqual(self.send().status_code, 200)
        db.update_folder(self.folder["id"], "Changed", "New prompt")
        db.update_session(self.sid, model="other-model", params_json='{"temperature":0}')
        job = db.get_generation(self.sid, db.last_message(self.sid)["id"])
        self.assertEqual(job["model"], "test-model")
        self.assertEqual(json.loads(job["messages_json"]), [{"role": "system", "content": "Be exact."}, {"role": "user", "content": "Hello"}])
        self.assertEqual(json.loads(job["params_json"])["temperature"], .95)
        self.assertEqual(db.get_session(self.sid)["title"], "Hello")

    def test_empty_chats_have_unique_titles_and_first_input_is_normalized(self):
        other = db.create_session(self.folder["id"])
        self.assertEqual(self.session["title"], f"New chat #{self.sid}")
        self.assertNotEqual(other["title"], self.session["title"])
        self.send("  First\n\tmessage   with  spacing  ")
        self.assertEqual(db.get_session(self.sid)["title"], "First message with spacing")
        self.finish()
        self.send("A different topic")
        self.assertEqual(db.get_session(self.sid)["title"], "First message with spacing")

    def test_automatic_title_truncates_unicode_and_spans_translation_chunks(self):
        db.set_folder_type(self.folder["id"], "translation")
        db.update_folder(self.folder["id"], "Translations", "", chunk_limit=5)
        self.send("한글🙂" * 30)
        title = db.get_session(self.sid)["title"]
        self.assertEqual(title, ("한글🙂" * 30)[:59] + "…")
        self.assertEqual(len(title), 60)

    def test_manual_title_survives_messages_and_blank_rename_restores_auto(self):
        path = f"/sessions/{self.sid}/rename"
        self.assertEqual(self.client.post(path, data={"title": "My project"}).status_code, 200)
        self.send("First input")
        self.finish()
        self.assertEqual(db.get_session(self.sid)["title"], "My project")
        self.assertEqual(self.client.post(path, data={"title": ""}).status_code, 200)
        self.assertEqual(db.get_session(self.sid)["title"], "First input")
        self.send("Second input")
        self.assertEqual(db.get_session(self.sid)["title"], "First input")
        self.client.post(path, data={"title": "New Chat"})
        db.init_db()
        self.assertEqual(db.get_session(self.sid)["title"], "New Chat")

    def test_note_title_and_delete_reset_title_and_update_date(self):
        db.set_folder_type(self.folder["id"], "memo")
        self.client.post(f"/sessions/{self.sid}/memo", data={"content": "A saved\n note"})
        self.assertEqual(db.get_session(self.sid)["title"], "A saved note")
        mid = db.last_message(self.sid)["id"]
        with db._lock, db._conn as conn:
            conn.execute("UPDATE sessions SET updated_at = '2000-01-01 00:00:00' WHERE id = ?", (self.sid,))
        response = self.client.post(f"/sessions/{self.sid}/messages/{mid}/delete")
        session = db.get_session(self.sid)
        self.assertEqual(session["title"], f"New chat #{self.sid}")
        self.assertGreater(session["updated_at"], "2000-01-01 00:00:00")
        self.assertIn(f'datetime="{session["updated_at"].replace(" ", "T")}+00:00"', response.text)

    def test_updated_date_reconciles_after_generation_and_settings(self):
        self.send()
        self.finish()
        job = db.list_generations(self.sid)[0]
        response = self.client.get(f'/sessions/{self.sid}/stream?since={job["user_message_id"]}')
        self.assertIn('id="folder-list" hx-swap-oob="innerHTML"', response.text)
        self.assertIn('class="session-modified"', response.text)
        response = self.client.post(f"/sessions/{self.sid}/model", data={"model": "other-model"})
        self.assertIn('class="session-modified"', response.text)

    def test_stop_and_retry_update_last_modification(self):
        self.send()
        for action in ("stop", "retry"):
            with db._lock, db._conn as conn:
                conn.execute("UPDATE sessions SET updated_at = '2000-01-01 00:00:00' WHERE id = ?", (self.sid,))
            response = self.client.post(f"/sessions/{self.sid}/{action}")
            self.assertEqual(response.status_code, 200)
            self.assertGreater(db.get_session(self.sid)["updated_at"], "2000-01-01 00:00:00")
            self.assertIn('class="session-modified"', response.text)

    def test_no_prompt_remains_no_prompt_in_viewer(self):
        db.update_folder(self.folder["id"], "Personal", "")
        self.send()
        job = self.finish()
        db.update_folder(self.folder["id"], "Personal", "DO NOT RETROACTIVELY ADD")
        response = self.client.get(f'/sessions/{self.sid}/messages/{job["assistant_message_id"]}/prompt')
        self.assertNotIn("DO NOT RETROACTIVELY ADD", response.text)
        self.assertIn("Exact messages saved", response.text)

    def test_overlapping_sends_are_atomic(self):
        def submit(_):
            try:
                db.enqueue_turns(self.sid, ["parallel"])
                return "accepted"
            except db.GenerationBusy:
                return "busy"
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(submit, range(6)))
        self.assertEqual(results.count("accepted"), 1)
        self.assertEqual(len(db.list_messages(self.sid)), 1)
        self.assertEqual(self.send("another").status_code, 409)

    def test_sse_preserves_whitespace_and_escapes_model_html(self):
        self.send()
        content = '<script>alert("unsafe")</script>\n\n    indented  code\nend'
        job = self.finish(content, "think\nagain")
        for _ in range(2):
            response = self.client.get(f'/sessions/{self.sid}/stream?since={job["user_message_id"]}')
            self.assertEqual(response.status_code, 200)
            self.assertIn("data:     indented  code", response.text)
            self.assertIn("&lt;script&gt;", response.text)
            self.assertNotIn('<script>alert', response.text)
        self.assertEqual(len([m for m in db.list_messages(self.sid) if m["role"] == "assistant"]), 1)
        self.assertEqual(_sse("content", "one\n\ntwo"), "event: content\ndata: one\ndata: \ndata: two\n\n")

    def test_stream_rejects_missing_and_foreign_user_ids(self):
        self.send()
        uid = db.last_message(self.sid)["id"]
        other = db.create_session(self.folder["id"])
        for path in (f"/sessions/{self.sid}/stream?since=0", f'/sessions/{other["id"]}/stream?since={uid}'):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_translation_chunks_are_lossless_bounded_and_paired(self):
        text = "First line.\n\n日本語です。 More words! " * 9
        chunks = _split_translation(text, 100)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(0 < len(chunk) <= 100 for chunk in chunks))
        db.set_folder_type(self.folder["id"], "translation")
        db.update_folder(self.folder["id"], "Translate", "Translate exactly", 100)
        self.assertEqual(self.send(text).status_code, 200)
        jobs = db.list_generations(self.sid)
        for job in jobs:
            request = json.loads(job["messages_json"])
            self.assertEqual([m["role"] for m in request], ["system", "user"])
            self.finish("Translated " + request[-1]["content"])
        visible = templating.chat_messages(self.sid)
        self.assertEqual([m["role"] for m in visible], ["user", "assistant"] * len(jobs))
        for index, job in enumerate(db.list_generations(self.sid)):
            response = self.client.get(f'/sessions/{self.sid}/messages/{job["assistant_message_id"]}/prompt')
            self.assertIn("Exact messages saved", response.text)
            self.assertEqual(visible[index * 2]["content"], json.loads(job["messages_json"])[-1]["content"])

    def test_translation_uses_one_event_source(self):
        db.set_folder_type(self.folder["id"], "translation")
        db.update_folder(self.folder["id"], "Translate", "", 100)
        response = self.send("a" * 800)
        self.assertEqual(response.text.count("sse-connect="), 1)
        self.assertEqual(response.text.count('data-generation-status="queued"'), 8)

    def test_translation_sse_activates_next_chunk_without_duplicate_ids(self):
        db.set_folder_type(self.folder["id"], "translation")
        db.update_folder(self.folder["id"], "Translate", "", 100)
        self.send("a" * 300)
        first = self.finish("First translation")
        response = self.client.get(f'/sessions/{self.sid}/stream?since={first["user_message_id"]}')
        done = next(event for event in response.text.split("\n\n") if event.startswith("event: done"))
        html = "\n".join(line.removeprefix("data: ") for line in done.splitlines()[1:])
        self.assertEqual(html.count("sse-connect="), 1)
        self.assertIn('hx-swap-oob="outerHTML:#stream-bubble-', html)
        parser = ElementIds()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))

    def test_actual_worker_completes_a_submitted_turn(self):
        def fake_stream(*args):
            yield {"type": "content", "text": "Background response"}
            yield {"type": "done", "usage": {"total_tokens": 2}}
        self.start.side_effect = START_WORKER
        with patch.object(ollama, "chat_stream", side_effect=fake_stream) as upstream:
            self.send()
            uid = db.list_generations(self.sid)[0]["user_message_id"]
            response = self.client.get(f"/sessions/{self.sid}/stream?since={uid}")
            self.assertIn("Background response", response.text)
            self.assertEqual(upstream.call_count, 1)
        self.assertEqual(db.list_generations(self.sid)[0]["status"], "completed")

    def test_page_has_unique_ids_and_active_navigation(self):
        self.send()
        response = self.client.get(f"/?session={self.sid}")
        parser = ElementIds()
        parser.feed(response.text)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertIn('aria-current="page"', response.text)
        self.assertIn('data-generation-status="queued"', response.text)

    def test_prompt_snapshot_survives_folder_conversion_and_message_deletion(self):
        self.send("Original input")
        job = self.finish("Original output")
        user_id = job["user_message_id"]
        # A later turn keeps its historical snapshot even if an earlier response is removed.
        self.send("Follow up")
        followup = self.finish("Follow-up answer")
        db.set_folder_type(self.folder["id"], "translation")
        db.delete_message(job["assistant_message_id"])
        response = self.client.get(f'/sessions/{self.sid}/messages/{followup["assistant_message_id"]}/prompt')
        self.assertIn("Original input", response.text)
        self.assertIn("Original output", response.text)
        self.assertIsNotNone(db.get_generation(self.sid, user_id))

    def test_stop_and_retry_ignore_late_worker_results(self):
        self.send()
        original = db.claim_generation(self.sid)
        db.update_generation(original, "Partial", "")
        response = self.client.post(f"/sessions/{self.sid}/stop")
        self.assertIn("Partial", response.text)
        self.assertIn("Retry with original settings", response.text)
        db.update_folder(self.folder["id"], "Changed", "Changed prompt")
        self.assertEqual(self.client.post(f"/sessions/{self.sid}/retry").status_code, 200)
        retry = db.claim_generation(self.sid)
        db.finish_generation(original, "Stale answer", "", None)
        db.interrupt_generations(self.sid, "Stale error", job=original)
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["status"], "running")
        self.assertEqual(original["messages_json"], retry["messages_json"])
        db.finish_generation(retry, "Correct answer", "", None)
        self.assertEqual(db.last_message(self.sid)["content"], "Correct answer")

    def test_worker_reports_failure_without_exposing_exception_html(self):
        self.send()
        job = db.claim_generation(self.sid)
        with patch.object(ollama, "chat_stream", side_effect=RuntimeError('<script>secret-key</script>')):
            generation._generate(job)
        response = self.client.get(f"/sessions/{self.sid}/messages")
        self.assertNotIn("secret-key", response.text)
        self.assertIn("Retry with original settings", response.text)

    def test_worker_persists_content_reasoning_and_usage(self):
        self.send()
        job = db.claim_generation(self.sid)
        chunks = [{"type": "reasoning", "text": "Think\n"}, {"type": "content", "text": "Answer\n  code"}, {"type": "done", "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}]
        with patch.object(ollama, "chat_stream", return_value=(chunk for chunk in chunks)):
            generation._generate(job)
        self.assertEqual(db.last_message(self.sid)["content"], "Answer\n  code")
        self.assertEqual(db.last_message(self.sid)["reasoning"], "Think\n")
        self.assertEqual(db.session_token_total(self.sid)["total"], 7)

    def test_retry_is_limited_to_latest_batch(self):
        self.send("Old failure")
        db.interrupt_generations(self.sid, "failed")
        self.send("New failure")
        db.interrupt_generations(self.sid, "failed")
        db.retry_generations(self.sid)
        self.assertEqual([j["status"] for j in db.list_generations(self.sid)], ["failed", "queued"])

    def test_restart_marks_interrupted_work_retryable(self):
        self.send()
        db.claim_generation(self.sid)
        db.recover_generations()
        self.assertEqual(db.list_generations(self.sid)[0]["status"], "failed")

    def test_settings_validation_and_partial_updates(self):
        original = json.loads(db.get_session(self.sid)["params_json"])
        response = self.client.post(f"/sessions/{self.sid}/model", data={"model": "other-model"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(db.get_session(self.sid)["params_json"]), original)
        for field, value in (("temperature", "nan"), ("temperature", 3), ("top_p", -1), ("seed", "oops"), ("max_tokens", 0), ("reasoning_effort", "invalid"), ("num_ctx", 8192)):
            response = self.client.post(f"/sessions/{self.sid}/model", data={"model": "other-model", field: value})
            self.assertEqual(response.status_code, 422, (field, response.text))
        response = self.client.post(f"/sessions/{self.sid}/model", data={"model": "other-model", "seed": 17, "temperature": .3}, headers={"HX-Target": "params-panel"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="model-selector" hx-swap-oob="true"', response.text)
        self.assertNotIn('id="params-panel" hx-swap-oob=', response.text)
        self.assertEqual(json.loads(db.get_session(self.sid)["params_json"])["seed"], 17)

    def test_model_missing_from_catalog_stays_selected(self):
        db.update_session(self.sid, model="removed-model")
        response = self.client.get(f"/?session={self.sid}")
        self.assertIn('value="removed-model" selected', response.text)

    def test_delete_active_chat_replaces_surface_but_inactive_does_not(self):
        other = db.create_session(self.folder["id"])
        response = self.client.post(f'/sessions/{other["id"]}/delete', data={"active_session_id": self.sid})
        self.assertNotIn('id="chat-surface"', response.text)
        response = self.client.post(f"/sessions/{self.sid}/delete", data={"active_session_id": self.sid})
        self.assertIn('id="chat-surface" hx-swap-oob="true"', response.text)
        self.assertNotIn('class="composer"', response.text)

    def test_delete_folder_cascades_and_clears_active_surface(self):
        self.send()
        response = self.client.post(f'/folders/{self.folder["id"]}/delete', data={"active_session_id": self.sid})
        self.assertIn('id="chat-surface" hx-swap-oob="true"', response.text)
        self.assertEqual(db.list_generations(self.sid), [])
        self.assertEqual(db.list_messages(self.sid), [])

    def test_memo_does_not_contact_models_or_erase_hidden_prompt(self):
        db.set_folder_type(self.folder["id"], "memo")
        with patch.object(ollama, "list_models", side_effect=AssertionError("No network for notes")):
            response = self.client.get(f"/?session={self.sid}")
            self.assertEqual(response.status_code, 200)
        response = self.client.post(f"/sessions/{self.sid}/memo", data={"content": "Keep this"})
        self.assertEqual(response.status_code, 200)
        self.start.assert_not_called()
        self.assertEqual(self.send().status_code, 403)
        self.client.post(f'/folders/{self.folder["id"]}/update', data={"name": "Notes"})
        self.assertEqual(db.get_folder(self.folder["id"])["system_prompt"], "Be exact.")

    def test_deleting_response_updates_usage_without_recreating_it(self):
        self.send()
        job = self.finish(usage={"total_tokens": 12})
        response = self.client.post(f'/sessions/{self.sid}/messages/{job["assistant_message_id"]}/delete')
        self.assertEqual(response.status_code, 200)
        self.assertIn("Tokens: 0", response.text)
        self.assertNotIn("sse-connect", response.text)


class OllamaTests(unittest.TestCase):
    def stream(self, lines):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="\n".join(lines)))
        client = httpx.Client(transport=transport, base_url="https://test.invalid/v1")
        with patch.object(ollama, "_client", return_value=client):
            return list(ollama.chat_stream("test-model", [], {}))

    def test_finish_chunk_content_and_usage_are_not_lost(self):
        chunks = self.stream([
            'data: {"choices":[{"delta":{"content":"last token","reasoning_content":"thinking"},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"total_tokens":4}}',
            'data: [DONE]',
        ])
        self.assertEqual(chunks[0], {"type": "reasoning", "text": "thinking"})
        self.assertEqual(chunks[1], {"type": "content", "text": "last token"})
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 4)

    def test_truncated_or_malformed_stream_is_not_success(self):
        for lines in ([ 'data: {"choices":[{"delta":{"content":"partial"}}]}' ], ['data: invalid'], ['data: {"error":"failed"}']):
            with self.assertRaises(RuntimeError):
                self.stream(lines)

    def test_seed_is_sent_at_top_level_and_unsupported_options_are_omitted(self):
        payload = ollama._build_payload("test-model", [], {"seed": 42, "num_ctx": 8192, "top_k": 40})
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("options", payload)


class MigrationTests(unittest.TestCase):
    def test_legacy_database_is_migrated_without_losing_messages(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            with sqlite3.connect(path) as conn:
                conn.executescript("""
                    CREATE TABLE folders (id INTEGER PRIMARY KEY, name TEXT, system_prompt TEXT, is_memo INTEGER, created_at TEXT);
                    CREATE TABLE sessions (id INTEGER PRIMARY KEY, folder_id INTEGER REFERENCES folders(id), title TEXT, model TEXT, params_json TEXT, created_at TEXT, updated_at TEXT);
                    CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER REFERENCES sessions(id), role TEXT, content TEXT, reasoning TEXT, model TEXT, reasoning_effort TEXT, created_at TEXT);
                    INSERT INTO folders VALUES (1, 'Old notes', 'Keep this prompt', 1, '2026-01-01');
                    INSERT INTO sessions VALUES (1, 1, 'Old chat', 'test-model', '{}', '2026-01-01', '2026-01-01');
                    INSERT INTO messages VALUES (1, 1, 'memo', 'A saved note', NULL, NULL, NULL, '2026-01-01');
                    INSERT INTO sessions VALUES (2, 1, 'New Chat', 'test-model', '{}', '2026-01-01', '2026-02-03');
                    INSERT INTO messages VALUES (2, 2, 'system', 'Not a title', NULL, NULL, NULL, '2026-01-01');
                    INSERT INTO messages VALUES (3, 2, 'user', '  First   old input  ', NULL, NULL, NULL, '2026-01-01');
                    INSERT INTO messages VALUES (4, 2, 'user', 'Later input', NULL, NULL, NULL, '2026-01-01');
                    INSERT INTO sessions VALUES (3, 1, NULL, 'test-model', '{}', '2026-01-01', '2026-01-01');
                    INSERT INTO messages VALUES (5, 3, 'memo', 'Old note', NULL, NULL, NULL, '2026-01-01');
                    INSERT INTO sessions VALUES (4, 1, '   ', 'test-model', '{}', '2026-01-01', '2026-01-01');
                """)
            with patch.object(db, "DB_PATH", path), patch.object(db, "_conn", None):
                try:
                    db.init_db()
                    db.init_db()  # Startup migration is idempotent.
                    self.assertEqual(db.get_folder(1)["type"], "memo")
                    self.assertEqual(db.get_folder(1)["system_prompt"], "Keep this prompt")
                    self.assertEqual(db.list_messages(1)[0]["content"], "A saved note")
                    self.assertEqual(db.list_generations(1), [])
                    self.assertEqual(db.get_session(1)["title"], "Old chat")
                    self.assertEqual(db.get_session(2)["title"], "First old input")
                    self.assertEqual(db.get_session(2)["updated_at"], "2026-02-03")
                    self.assertEqual(db.get_session(3)["title"], "Old note")
                    self.assertEqual(db.get_session(4)["title"], "New chat #4")
                finally:
                    db._conn.close()


if __name__ == "__main__":
    unittest.main()
