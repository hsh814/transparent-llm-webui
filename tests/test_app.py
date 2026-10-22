"""Offline regressions. Run: uv run python -m unittest discover -s tests -v."""

import asyncio
import base64
import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OLLAMA_API_KEY", "test-key")

import httpx
from fastapi.testclient import TestClient
from PIL import Image

import db
import generation
import inputs
import ollama
import payloads
import templating
from app import app
from routers.chat import _split_translation, _sse, stream_message

START_WORKER = generation.start


def paper_pdf(page_count=2):
    """Real PDF: selectable table text plus a vector figure, then image-only pages."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" + " ".join(f"{4 + i * 2} 0 R" for i in range(page_count)) +
         f"] /Count {page_count} >>").encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i in range(page_count):
        stream = b"1 0 0 rg 20 20 100 60 re f\n"
        if i == 0:
            stream += b"BT /F1 14 Tf 20 250 Td (Table 1: Method A = 42) Tj ET\n"
        objects.extend([
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 300] "
             f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + i * 2} 0 R >>").encode(),
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
        ])
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010} 00000 n \n".encode())
    result.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    return bytes(result)


def image_png():
    with BytesIO() as output, Image.new("RGB", (32, 16), (10, 90, 180)) as image:
        image.save(output, format="PNG")
        return output.getvalue()


def search_call(query="model-selected query", call_id="search-1", max_results=2):
    return {"id": call_id, "type": "function", "function": {
        "name": "web_search", "arguments": json.dumps({"query": query, "max_results": max_results}),
    }}


def search_stream(model, messages, params, *, tools=None):
    if messages[-1]["role"] != "tool":
        yield {"type": "content", "text": "Checking the sources."}
        yield {"type": "done", "finish_reason": "tool_calls", "tool_calls": [search_call()],
               "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}
    else:
        yield {"type": "content", "text": "First answer"}
        yield {"type": "done", "finish_reason": "stop",
               "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37}}


def legacy_payload_database(path):
    """Pre-reference schema with copied inputs, historical variants and active work."""
    url = "data:image/png;base64," + base64.b64encode(image_png()).decode()
    attachments = [{"name": "figure.png", "media_type": "image/png", "data_url": url}]
    search = {"status": "completed", "query": "paper", "max_results": 5,
              "results": [{"title": "Paper", "url": "https://example.org/paper", "content": "Saved result"}]}
    content = inputs.build_content("Figure question", attachments)
    content.append({"type": "text", "text": "Web search reference material:\n" + json.dumps(search["results"])})
    messages = [{"role": "system", "content": "Saved system"}, {"role": "user", "content": content}]
    current = {"model": "test-model", "params_json": '{"temperature":0.25}',
               "messages_json": json.dumps(messages), "search_json": json.dumps(search),
               "system_prompt_hash": None}
    historical = {**current, "model": "older-model", "messages_json": json.dumps([
        {"role": "system", "content": "Historical system"}, {"role": "user", "content": content},
    ])}
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE folders (
              id INTEGER PRIMARY KEY, name TEXT, system_prompt TEXT, type TEXT,
              chunk_limit INTEGER, pinned INTEGER, created_at TEXT);
            CREATE TABLE sessions (
              id INTEGER PRIMARY KEY, folder_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
              title TEXT, auto_title INTEGER, model TEXT, params_json TEXT,
              created_at TEXT, updated_at TEXT, params_updated_at TEXT);
            CREATE TABLE messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
              role TEXT, content TEXT, reasoning TEXT, model TEXT, reasoning_effort TEXT,
              prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
              attachments_json TEXT NOT NULL DEFAULT '[]', api_content_json TEXT,
              system_prompt_hash TEXT, created_at TEXT DEFAULT '2026-01-01');
            CREATE TABLE generations (
              user_message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
              session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
              model TEXT, params_json TEXT, messages_json TEXT, search_json TEXT,
              system_prompt_hash TEXT, batch_id INTEGER, attempt INTEGER, status TEXT,
              content TEXT, reasoning TEXT, error TEXT,
              assistant_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL);
            CREATE TABLE response_variants (
              assistant_message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
              user_message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE, request_json TEXT);
            INSERT INTO folders VALUES (1, 'Legacy', 'Current system', 'chat', NULL, 0, '2026-01-01');
            INSERT INTO sessions VALUES (1, 1, 'Original', 0, 'current-model', '{}', '2026-01-01', '2026-02-03', NULL);
            INSERT INTO sessions VALUES (2, 1, 'Fork', 0, 'current-model', '{}', '2026-01-01', '2026-02-04', NULL);
        """)
        for mid, sid, role, text in (
            (1, 1, "user", "Figure question"), (2, 1, "assistant", "Historical answer"),
            (3, 1, "assistant", "Selected answer"), (4, 1, "assistant", "Unrecorded legacy answer"),
            (10, 2, "user", "Figure question"), (11, 2, "assistant", "Selected answer"),
            (20, 1, "user", "Pending question"), (5000, 1, "memo", "Previously deleted"),
        ):
            conn.execute("INSERT INTO messages (id, session_id, role, content, attachments_json, api_content_json)"
                         " VALUES (?, ?, ?, ?, ?, ?)",
                         (mid, sid, role, text, json.dumps(attachments if mid in (1, 10) else []),
                          json.dumps(content) if mid in (1, 10) else None))
        conn.execute("DELETE FROM messages WHERE id = 5000")
        for uid, sid, aid in ((1, 1, 3), (10, 2, 11), (20, 1, None)):
            row = {**current, "user_message_id": uid, "session_id": sid, "batch_id": uid,
                   "attempt": 7 if uid == 20 else 2, "status": "running" if uid == 20 else "completed",
                   "content": "Partial" if uid == 20 else "Selected answer", "reasoning": "",
                   "error": None, "assistant_message_id": aid}
            conn.execute(f"INSERT INTO generations ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                         tuple(row.values()))
            if aid is not None:
                conn.execute("INSERT INTO response_variants VALUES (?, ?, ?)", (aid, uid, json.dumps(row)))
        conn.execute("INSERT INTO response_variants VALUES (2, 1, ?)", (json.dumps({
            **historical, "user_message_id": 1, "session_id": 1, "assistant_message_id": 2,
        }),))
        conn.execute("INSERT INTO response_variants VALUES (4, 1, NULL)")
    return {"messages": messages, "search": search, "url": url,
            "historical_messages": json.loads(historical["messages_json"])}


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

    def prompt_messages(self, response):
        self.assertEqual(response.status_code, 200)
        text = response.text.split('<pre id="prompt-messages">', 1)[1].split("</pre>", 1)[0]
        return json.loads(unescape(text))

    def payload_stats(self):
        with db._lock:
            return tuple(db._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(body)), 0) FROM payloads",
            ).fetchone())

    def test_tool_events_stream_before_the_final_model_output(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Find a paper", "web_search": "true"})
        job = db.claim_generation(self.sid)
        db.record_tool_calls(job, {"role": "assistant", "content": "", "tool_calls": [search_call()]}, "", None)
        db.start_tool(job, "search-1")

        async def observe_transition():
            request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
            response = stream_message(request, self.sid, job["user_message_id"])
            events = response.body_iterator

            async def until(name, text):
                async for event in events:
                    if event.startswith(f"event: {name}\n") and text in event:
                        return event
                self.fail(f"Missing {name} event containing {text!r}")

            try:
                await until("trace", "model-selected query")
                db.finish_tool(job, "search-1", {"query": "model-selected query", "max_results": 2,
                    "results": [{"title": "A source", "url": "https://example.org", "content": "A result"}],
                    "fetched_at": "2026-10-22T13:00:00Z"})
                db.update_generation(job, "Answer is arriving", "")
                await until("trace", "https://example.org")
                await until("content", "Answer is arriving")
            finally:
                await events.aclose()

        asyncio.run(asyncio.wait_for(observe_transition(), 3))

    def test_image_only_input_survives_followup_and_prompt_inspection(self):
        original = image_png()
        response = self.client.post(f"/sessions/{self.sid}/send",
                                    files=[("files", ("diagram.png", original, "image/png"))])
        self.assertEqual(response.status_code, 200, response.text)
        job = db.list_generations(self.sid)[0]
        parts = job["messages"][-1]["content"]
        image_part = next(part for part in parts if part["type"] == "image_url")
        self.assertEqual(base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]), original)
        saved = db.last_message(self.sid)["attachments"][0]
        self.assertEqual(saved["sha256"], hashlib.sha256(original).hexdigest())
        prompt = self.client.get(f'/sessions/{self.sid}/messages/{job["user_message_id"]}/prompt')
        self.assertEqual(self.prompt_messages(prompt)[-1]["content"], parts)
        self.finish()
        self.send("Explain the earlier diagram")
        followup = db.list_generations(self.sid)[-1]
        self.assertEqual(followup["messages"][1]["content"], parts)

    def test_pdf_preserves_page_images_tables_and_pages_without_text(self):
        response = self.client.post(f"/sessions/{self.sid}/send", data={"content": "Read the paper"},
                                    files=[("files", ("paper.pdf", paper_pdf(), "application/pdf"))])
        self.assertEqual(response.status_code, 200, response.text)
        attachment = db.last_message(self.sid)["attachments"][0]
        self.assertEqual([p["number"] for p in attachment["pages"]], [1, 2])
        self.assertIn("Method A = 42", attachment["pages"][0]["text"])
        self.assertEqual(attachment["pages"][1]["text"], "")
        for page in attachment["pages"]:
            with Image.open(BytesIO(base64.b64decode(page["data_url"].split(",", 1)[1]))) as image:
                # Both pages' figures survived rasterization, including the textless page.
                self.assertEqual(image.size, (480, 600))
                red, green, blue = image.getpixel((100, 500))
                self.assertGreater(red, 240)
                self.assertLess(green, 15)
                self.assertLess(blue, 15)
        parts = db.list_generations(self.sid)[0]["messages"][-1]["content"]
        sent_images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
        self.assertEqual(sent_images, [p["data_url"] for p in attachment["pages"]])
        job = self.finish()
        prompt = self.client.get(f'/sessions/{self.sid}/messages/{job["assistant_message_id"]}/prompt')
        self.assertEqual(self.prompt_messages(prompt)[-1]["content"], parts)

    def test_attachment_rejection_is_atomic_and_never_truncates_pdfs(self):
        for files in (
            [("files", ("fake.png", b"not an image", "image/png"))],
            [("files", ("broken.pdf", b"%PDF-broken", "application/pdf"))],
            [("files", ("empty.png", b"", "image/png"))],
            [("files", ("many.pdf", paper_pdf(21), "application/pdf"))],
            [("files", ("a.png", image_png(), "image/png"))] * 5,
        ):
            with self.subTest(names=[f[1][0] for f in files]):
                response = self.client.post(f"/sessions/{self.sid}/send", data={"content": "Read"}, files=files)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(db.list_messages(self.sid), [])
                self.assertEqual(db.list_generations(self.sid), [])
        with patch.object(inputs, "MAX_TOTAL_BYTES", len(image_png())):
            response = self.client.post(f"/sessions/{self.sid}/send",
                                        files=[("files", ("a.png", image_png(), "image/png"))] * 2)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(db.list_messages(self.sid), [])
        self.start.assert_not_called()

    def test_failed_tool_is_visible_and_retried_without_a_second_model_request(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Latest paper", "web_search": "true"})
        with patch.object(ollama, "web_search", side_effect=httpx.ConnectError("secret detail")), \
                patch.object(ollama, "chat_stream", side_effect=search_stream) as model:
            generation._generate(db.claim_generation(self.sid))
            self.assertEqual(model.call_count, 1)
        job = db.list_generations(self.sid)[0]
        self.assertEqual(job["status"], "failed")
        self.assertEqual([event["kind"] for event in job["trace"]], ["assistant", "tool"])
        self.assertEqual(job["trace"][-1]["status"], "failed")
        self.assertEqual(json.loads(job["trace"][0]["message"]["tool_calls"][0]["function"]["arguments"])["query"],
                         "model-selected query")
        prompt = self.client.get(f'/sessions/{self.sid}/messages/{job["user_message_id"]}/prompt')
        self.assertNotIn("secret detail", prompt.text)
        self.assertEqual(self.prompt_messages(prompt)[1]["content"], "Latest paper")
        self.assertTrue(db.retry_generations(self.sid))
        with patch.object(ollama, "web_search", return_value=[]) as search, \
                patch.object(ollama, "chat_stream", side_effect=search_stream) as model:
            generation._generate(db.claim_generation(self.sid))
            search.assert_called_once_with("model-selected query", max_results=2)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(model.call_args.args[1][-1]["role"], "tool")
        completed = db.list_generations(self.sid)[0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(json.loads(completed["trace"][-1]["message"]["content"])["results"], [])

    def test_search_requires_opt_in_and_chat_mode(self):
        self.send("No search")
        with patch.object(ollama, "web_search", side_effect=AssertionError("No opt-in")), \
                patch.object(ollama, "chat_stream", side_effect=search_stream):
            generation._generate(db.claim_generation(self.sid))
        self.assertEqual(db.list_generations(self.sid)[0]["status"], "failed")
        for mode in ("translation", "memo"):
            folder = db.create_folder("Other mode", "", mode)
            sid = db.create_session(folder["id"])["id"]
            for options in ({"data": {"content": "Text", "web_search": "true"}},
                            {"files": [("files", ("a.png", image_png(), "image/png"))]}):
                response = self.client.post(f"/sessions/{sid}/send", **options)
                self.assertIn(response.status_code, (400, 403))
                self.assertEqual(db.list_messages(sid), [])

    def test_search_permission_does_not_force_a_tool_call(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Hello", "web_search": "true"})
        with patch.object(ollama, "web_search") as search, patch.object(ollama, "chat_stream", return_value=(chunk for chunk in [
            {"type": "content", "text": "Hello back"}, {"type": "done"},
        ])) as model:
            generation._generate(db.claim_generation(self.sid))
            search.assert_not_called()
            self.assertEqual(model.call_args.kwargs["tools"][0]["function"]["name"], "web_search")
        job = db.list_generations(self.sid)[0]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["trace"], [])
        self.assertEqual(job["messages"][-1], {"role": "user", "content": "Hello"})

    def test_search_results_and_multimodal_request_are_reused_on_retry(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Compare this", "web_search": "true"},
                         files=[("files", ("figure.png", image_png(), "image/png"))])
        results = [{"title": "Source", "url": "https://example.org/paper", "content": "Reported value: 42"}]

        def fail_after_search(model, messages, params, *, tools=None):
            if messages[-1]["role"] == "tool":
                raise httpx.ConnectError("offline")
            yield from search_stream(model, messages, params, tools=tools)

        with patch.object(ollama, "web_search", return_value=results), \
                patch.object(ollama, "chat_stream", side_effect=fail_after_search):
            generation._generate(db.claim_generation(self.sid))
        failed = db.list_generations(self.sid)[0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(json.loads(failed["trace"][-1]["message"]["content"])["results"], results)
        self.assertEqual([message["role"] for message in failed["messages"]], ["system", "user", "assistant", "tool"])
        parts = failed["messages"][1]["content"]
        self.assertNotIn("Reported value: 42", json.dumps(parts))
        self.assertEqual(parts, db.visible_messages(self.sid)[0]["api_content"])
        db.update_folder(self.folder["id"], "Changed", "Do something else.")
        self.assertTrue(db.retry_generations(self.sid))
        with patch.object(ollama, "web_search", side_effect=AssertionError("Do not repeat saved search")), \
                patch.object(ollama, "chat_stream", side_effect=search_stream):
            generation._generate(db.claim_generation(self.sid))
        retried = db.list_generations(self.sid)[0]
        self.assertEqual(retried["status"], "completed")
        self.assertEqual(retried["messages"], failed["messages"])
        self.assertEqual(retried["trace"], failed["trace"])
        self.assertEqual(db.visible_messages(self.sid)[-1]["total_tokens"], 50)
        self.send("Follow up without a fresh search")
        followup = db.list_generations(self.sid)[-1]
        self.assertEqual(followup["tools"], [])
        self.assertEqual([message["role"] for message in followup["messages"]],
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(followup["messages"][1]["content"], parts)
        self.assertEqual(followup["messages"][3], retried["messages"][3])

    def test_cancelled_search_cannot_replace_retry_or_contact_model_again(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Find paper", "web_search": "true"})
        stale = db.claim_generation(self.sid)
        during_retry = {}

        def late_search(*args, **kwargs):
            db.interrupt_generations(self.sid, "Stopped", "cancelled")
            db.retry_generations(self.sid)
            during_retry["size"] = self.payload_stats()
            return [{"title": "Stale", "url": "https://example.org/old", "content": "Old result"}]

        with patch.object(ollama, "web_search", side_effect=late_search), \
                patch.object(ollama, "chat_stream", side_effect=search_stream) as model:
            generation._generate(stale)
            self.assertEqual(model.call_count, 1)
        current = db.list_generations(self.sid)[0]
        self.assertEqual(current["attempt"], stale["attempt"] + 1)
        self.assertEqual(current["status"], "queued")
        self.assertNotIn("tool", [message["role"] for message in current["messages"]])
        self.assertNotIn("https://example.org/old", json.dumps(current))
        self.assertEqual(self.payload_stats(), during_retry["size"])

    def test_event_forks_keep_exact_tool_boundaries_and_resume_only_missing_results(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Compare two sources", "web_search": "true"})
        calls = [search_call("first query", "call-1"), search_call("second query", "call-2")]
        requests = []

        def model(model, messages, params, *, tools=None):
            requests.append({"messages": messages, "tools": tools})
            if messages[-1]["role"] != "tool":
                yield {"type": "done", "finish_reason": "tool_calls", "tool_calls": calls}
            else:
                yield from search_stream(model, messages, params, tools=tools)

        def results(query, **kwargs):
            return [{"title": query, "url": "https://example.org", "content": f"Evidence for {query}"}]

        with patch.object(ollama, "chat_stream", side_effect=model), \
                patch.object(ollama, "web_search", side_effect=results):
            generation._generate(db.claim_generation(self.sid))
        source = db.list_generations(self.sid)[0]
        uid = source["user_message_id"]
        self.assertEqual(source["status"], "completed")
        self.assertEqual([event["kind"] for event in source["trace"]], ["assistant", "tool", "tool"])
        first_prompt = self.client.get(
            f'/sessions/{self.sid}/messages/{uid}/events/0/prompt?trace_id={source["trace_id"]}')
        self.assertEqual(self.prompt_messages(first_prompt), requests[0]["messages"])
        final_prompt = self.client.get(
            f'/sessions/{self.sid}/messages/{source["assistant_message_id"]}/prompt')
        self.assertEqual(self.prompt_messages(final_prompt), requests[1]["messages"])
        self.assertEqual([message["role"] for message in requests[1]["messages"]],
                         ["system", "user", "assistant", "tool", "tool"])

        user_fork = db.fork_session(self.sid, uid)
        user_job = db.list_generations(user_fork["id"])[0]
        self.assertEqual(user_job["trace"], [])
        self.assertEqual(user_job["messages"], source["base_request"]["messages"])

        call_fork = db.fork_session(self.sid, uid, event_index=0, trace_id=source["trace_id"])
        call_job = db.list_generations(call_fork["id"])[0]
        self.assertEqual(call_job["trace"], source["trace"][:1])
        self.assertEqual(call_job["tools"], [])
        blocked = self.client.post(f'/sessions/{call_fork["id"]}/send', data={"content": "Skip ahead"})
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(len(db.visible_messages(call_fork["id"])), 1)
        db.request_response(call_fork["id"], call_job["user_message_id"], web_search=False)
        with patch.object(ollama, "web_search") as search, patch.object(ollama, "chat_stream") as upstream:
            generation._generate(db.claim_generation(call_fork["id"]))
            search.assert_not_called()
            upstream.assert_not_called()
        self.assertEqual(db.list_generations(call_fork["id"])[0]["status"], "failed")

        result_fork = db.fork_session(self.sid, uid, event_index=1, trace_id=source["trace_id"])
        result_job = db.list_generations(result_fork["id"])[0]
        self.assertEqual(result_job["trace"], source["trace"][:2])
        self.assertEqual(result_job["status"], "idle")
        self.assertNotIn("Evidence for second query", json.dumps(result_job["messages"]))
        db.request_response(result_fork["id"], result_job["user_message_id"], web_search=True)
        with patch.object(ollama, "web_search", side_effect=results) as search, \
                patch.object(ollama, "chat_stream", side_effect=model):
            generation._generate(db.claim_generation(result_fork["id"]))
            search.assert_called_once_with("second query", max_results=2)
        completed = db.list_generations(result_fork["id"])[0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["messages"][:4], source["messages"][:4])
        self.assertEqual(json.loads(completed["messages"][-1]["content"])["results"],
                         json.loads(source["messages"][-1]["content"])["results"])
        self.assertEqual(db.get_generation(self.sid, uid)["trace"], source["trace"])

        all_results_fork = db.fork_session(self.sid, uid, event_index=2, trace_id=source["trace_id"])
        all_job = db.list_generations(all_results_fork["id"])[0]
        self.assertEqual(all_job["trace_id"], source["trace_id"])
        db.request_response(all_results_fork["id"], all_job["user_message_id"], web_search=False)
        with patch.object(ollama, "web_search") as search, patch.object(ollama, "chat_stream", side_effect=model):
            generation._generate(db.claim_generation(all_results_fork["id"]))
            search.assert_not_called()
        self.assertEqual(db.list_generations(all_results_fork["id"])[0]["status"], "completed")
        for index, trace_id in ((99, source["trace_id"]), (1, "not-the-visible-trace")):
            response = self.client.post(f"/sessions/{self.sid}/messages/{uid}/events/{index}/fork",
                                        data={"trace_id": trace_id})
            self.assertIn(response.status_code, (400, 404, 409))

    def test_regeneration_selects_its_own_trace_and_uses_current_search_permission(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Research this", "web_search": "true"})
        with patch.object(ollama, "chat_stream", side_effect=search_stream), \
                patch.object(ollama, "web_search", return_value=[]):
            generation._generate(db.claim_generation(self.sid))
        original = db.list_generations(self.sid)[0]
        response = self.client.post(
            f'/sessions/{self.sid}/messages/{original["assistant_message_id"]}/regenerate',
            data={"web_search": "false"})
        self.assertEqual(response.status_code, 200)
        active = db.list_generations(self.sid)[0]
        self.assertEqual(active["trace"], [])
        self.assertEqual(active["tools"], [])
        self.assertEqual(db.get_message_generation(self.sid, original["assistant_message_id"])["trace"],
                         original["trace"])
        self.finish("A different answer without search")
        alternate = db.list_generations(self.sid)[0]
        self.assertEqual(alternate["trace"], [])
        db.select_response_variant(self.sid, original["user_message_id"], original["assistant_message_id"])
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["trace_id"], original["trace_id"])
        db.select_response_variant(self.sid, original["user_message_id"], alternate["assistant_message_id"])
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["trace"], [])

    def test_model_cannot_search_indefinitely_across_tool_rounds(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Research", "web_search": "true"})

        def keep_searching(model, messages, params, *, tools=None):
            index = sum(message["role"] == "tool" for message in messages) + 1
            yield {"type": "done", "finish_reason": "tool_calls",
                   "tool_calls": [search_call(f"query {index}", f"call-{index}")]}

        with patch.object(ollama, "chat_stream", side_effect=keep_searching), \
                patch.object(ollama, "web_search", return_value=[]) as search:
            generation._generate(db.claim_generation(self.sid))
            self.assertEqual(search.call_count, 8)
        job = db.list_generations(self.sid)[0]
        self.assertEqual(job["status"], "failed")
        self.assertIsNone(job["assistant_message_id"])
        self.assertEqual(sum(event["kind"] == "tool" and event["status"] == "completed"
                             for event in job["trace"]), 8)

    def test_closed_tool_batches_pair_every_call_before_a_new_user_turn(self):
        for close in ("stop", "restart"):
            with self.subTest(close=close):
                sid = db.create_session(self.folder["id"], model="test-model")["id"]
                self.client.post(f"/sessions/{sid}/send", data={"content": "Research", "web_search": "true"})
                job = db.claim_generation(sid)
                calls = [search_call(f"query {index}", f"call-{index}") for index in (1, 2, 3)]
                db.record_tool_calls(job, {"role": "assistant", "content": "", "tool_calls": calls}, "", None)
                db.start_tool(job, "call-1")
                db.finish_tool(job, "call-1", {"query": "query 1", "max_results": 2,
                                              "results": [], "fetched_at": "2026-10-22T13:00:00Z"})
                db.start_tool(job, "call-2")
                if close == "stop":
                    db.interrupt_generations(sid, "Stopped by user", "cancelled")
                else:
                    db.recover_generations()
                saved = db.list_generations(sid)[0]
                self.assertEqual([event["status"] for event in saved["trace"]],
                                 ["completed", "completed", "failed", "failed"])
                self.assertEqual([message["tool_call_id"] for message in saved["messages"]
                                  if message["role"] == "tool"], ["call-1"])
                self.assertEqual(self.client.post(f"/sessions/{sid}/send",
                                                 data={"content": "Continue without searching"}).status_code, 200)
                followup = db.list_generations(sid)[-1]
                self.assertEqual([message["role"] for message in followup["messages"]],
                                 ["system", "user", "assistant", "tool", "tool", "tool", "user"])
                self.assertEqual([message["tool_call_id"] for message in followup["messages"][3:6]],
                                 ["call-1", "call-2", "call-3"])
                for message in followup["messages"][4:6]:
                    self.assertEqual(json.loads(message["content"]), {"error": saved["error"]})
                self.assertEqual(followup["tools"], [])

    def test_prompt_json_matches_outgoing_messages_after_settings_change(self):
        for mode in ("chat", "translation"):
            with self.subTest(mode=mode):
                folder = db.create_folder("Fidelity", "  System\r\n\t日本語 <>& \"quoted\"  ", mode, 10000)
                session = db.create_session(folder["id"], model="test-model")
                sid = session["id"]
                db.add_message(sid, "memo", "Private note: never send")
                db.add_message(sid, "user", "Earlier question")
                db.add_message(sid, "assistant", "Earlier answer")
                source = "Literal </pre><script>unsafe()</script>\n\t  indented\r\nEnd"
                self.client.post(f"/sessions/{sid}/send", data={"content": source})
                captured = []

                def respond(request):
                    captured.append(json.loads(request.content))
                    return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

                with patch.object(ollama, "_client", side_effect=lambda: httpx.Client(
                    base_url="https://example.test/v1", transport=httpx.MockTransport(respond)
                )):
                    generation._generate(db.claim_generation(sid))
                job = db.list_generations(sid)[0]
                self.assertEqual(job["status"], "completed")
                db.update_folder(folder["id"], "Changed", "Different system prompt")
                db.set_folder_type(folder["id"], "memo")
                db.update_session(sid, model="other-model", params_json='{"temperature":0}')
                response = self.client.get(f'/sessions/{sid}/messages/{job["assistant_message_id"]}/prompt')
                displayed = self.prompt_messages(response)
                self.assertEqual(displayed, captured[0]["messages"])
                self.assertEqual(displayed[0]["content"], folder["system_prompt"])
                self.assertEqual(displayed[-1]["content"], source)
                self.assertEqual([m["role"] for m in displayed],
                                 ["system", "user", "assistant", "user"] if mode == "chat" else ["system", "user"])

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
        self.assertEqual(job["messages"], [{"role": "system", "content": "Be exact."}, {"role": "user", "content": "Hello"}])
        self.assertEqual(job["params"]["temperature"], .95)
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
        self.assertEqual(self.prompt_messages(response), [{"role": "user", "content": "Hello"}])

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
            request = job["messages"]
            self.assertEqual([m["role"] for m in request], ["system", "user"])
            self.finish("Translated " + request[-1]["content"])
        page = self.client.get(f"/sessions/{self.sid}/messages")
        rendered = ElementIds()
        rendered.feed(page.text)
        expected = [f"message-{mid}" for job in db.list_generations(self.sid)
                    for mid in (job["user_message_id"], job["assistant_message_id"])]
        self.assertEqual([element_id for element_id in rendered.ids if element_id.startswith("message-")], expected)
        for job in db.list_generations(self.sid):
            response = self.client.get(f'/sessions/{self.sid}/messages/{job["assistant_message_id"]}/prompt')
            self.assertEqual(self.prompt_messages(response), job["messages"])

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
        def fake_stream(*args, **kwargs):
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
        db.update_folder(self.folder["id"], "Changed", "Changed prompt")
        self.assertEqual(self.client.post(f"/sessions/{self.sid}/retry").status_code, 200)
        retry = db.claim_generation(self.sid)
        db.finish_generation(original, "Stale answer", "", None)
        db.interrupt_generations(self.sid, "Stale error", job=original)
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["status"], "running")
        self.assertEqual(original["messages"], retry["messages"])
        db.finish_generation(retry, "Correct answer", "", None)
        self.assertEqual(db.last_message(self.sid)["content"], "Correct answer")

    def test_worker_reports_failure_without_exposing_exception_html(self):
        self.send()
        job = db.claim_generation(self.sid)
        with patch.object(ollama, "chat_stream", side_effect=RuntimeError('<script>secret-key</script>')):
            generation._generate(job)
        response = self.client.get(f"/sessions/{self.sid}/messages")
        self.assertNotIn("secret-key", response.text)

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

    def test_selecting_old_response_never_reuses_a_cancelled_attempt(self):
        self.send("Question")
        original = self.finish("Original")
        db.regenerate_response(self.sid, original["assistant_message_id"])
        stale = db.claim_generation(self.sid)
        db.interrupt_generations(self.sid, "Stopped", "cancelled")
        db.select_response_variant(self.sid, original["user_message_id"], original["assistant_message_id"])
        db.regenerate_response(self.sid, original["assistant_message_id"])
        fresh = db.claim_generation(self.sid)
        self.assertGreater(fresh["attempt"], stale["attempt"])
        db.finish_generation(stale, "Stale", "", None)
        self.assertEqual(db.response_variant_state(self.sid, original["user_message_id"])["count"], 1)
        db.finish_generation(fresh, "Fresh", "", None)
        self.assertEqual(db.last_message(self.sid)["content"], "Fresh")

    def test_existing_response_snapshot_backfill_is_idempotent(self):
        self.send("Existing input")
        original = self.finish("Existing answer")
        modified = db.get_session(self.sid)["updated_at"]
        with db._lock, db._conn as conn:
            # Simulate upgrading a database that predates response-variant metadata.
            conn.execute("DELETE FROM response_variants")
        db.init_db()
        db.init_db()
        state = db.response_variant_state(self.sid, original["user_message_id"])
        self.assertEqual(state["ids"], [original["assistant_message_id"]])
        self.assertEqual(db.get_session(self.sid)["updated_at"], modified)
        self.assertEqual(db.get_message_generation(self.sid, original["assistant_message_id"])["messages"],
                         original["messages"])

    def test_request_after_deleting_response_reuses_input_without_duplicate_user(self):
        self.send("Keep this question")
        original = self.finish("First answer")
        self.client.post(f'/sessions/{self.sid}/messages/{original["assistant_message_id"]}/delete')
        db.update_folder(self.folder["id"], "Changed", "Different system")
        db.update_session(self.sid, model="other-model", params_json='{"temperature":0}')
        response = self.client.post(f'/sessions/{self.sid}/messages/{original["user_message_id"]}/respond')
        self.assertEqual(response.status_code, 200, response.text)
        queued = db.get_generation(self.sid, original["user_message_id"])
        self.assertEqual(queued["messages"], original["messages"])
        self.assertEqual(queued["model"], original["model"])
        self.assertEqual(queued["params"], original["params"])
        self.assertEqual([m["content"] for m in db.list_messages(self.sid)], ["Keep this question"])
        regenerated = self.finish("Replacement answer")
        self.assertEqual([m["content"] for m in db.visible_messages(self.sid)], ["Keep this question", "Replacement answer"])
        self.assertNotEqual(regenerated["assistant_message_id"], original["assistant_message_id"])

    def test_legacy_user_can_request_response_without_resending(self):
        user = db.add_message(self.sid, "user", "An unanswered legacy question")
        response = self.client.post(f'/sessions/{self.sid}/messages/{user["id"]}/respond')
        self.assertEqual(response.status_code, 200, response.text)
        job = db.get_generation(self.sid, user["id"])
        self.assertEqual(job["messages"], [
            {"role": "system", "content": "Be exact."},
            {"role": "user", "content": "An unanswered legacy question"},
        ])
        self.assertEqual(len(db.list_messages(self.sid)), 1)
        self.finish("New answer")
        self.assertEqual(db.last_message(self.sid)["content"], "New answer")

    def test_regeneration_preserves_versions_and_selected_history(self):
        self.send("Question")
        original = self.finish("First version", usage={"total_tokens": 3})
        db.update_folder(self.folder["id"], "Changed", "Changed system")
        db.update_session(self.sid, model="other-model", params_json='{"temperature":0}')
        response = self.client.post(f'/sessions/{self.sid}/messages/{original["assistant_message_id"]}/regenerate')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("First version", response.text)
        self.assertEqual(db.last_message(self.sid)["content"], "First version")
        queued = db.get_generation(self.sid, original["user_message_id"])
        self.assertEqual((queued["messages"], queued["model"], queued["params"]),
                         (original["messages"], original["model"], original["params"]))
        self.assertEqual(db.get_message_generation(self.sid, original["assistant_message_id"])["status"], "completed")
        second = self.finish("Second version", usage={"total_tokens": 5})
        state = db.response_variant_state(self.sid, original["user_message_id"])
        self.assertEqual(state["ids"], [original["assistant_message_id"], second["assistant_message_id"]])
        self.assertEqual(state["index"], 2)
        self.assertEqual(db.session_token_total(self.sid)["total"], 8)
        selected = self.client.post(
            f'/sessions/{self.sid}/messages/{original["user_message_id"]}/variants/{original["assistant_message_id"]}/select',
        )
        self.assertEqual(selected.status_code, 200, selected.text)
        self.assertEqual(db.last_message(self.sid)["id"], original["assistant_message_id"])
        self.assertEqual([m["content"] for m in db.visible_messages(self.sid)], ["Question", "First version"])
        for response_id in (original["assistant_message_id"], second["assistant_message_id"]):
            prompt = self.client.get(f"/sessions/{self.sid}/messages/{response_id}/prompt")
            self.assertEqual(self.prompt_messages(prompt), original["messages"])
        self.send("Follow up")
        followup = db.list_generations(self.sid)[-1]
        sent = followup["messages"]
        self.assertEqual([m["content"] for m in sent], ["Changed system", "Question", "First version", "Follow up"])
        self.finish("Follow-up answer")
        before = db.visible_messages(self.sid)
        forbidden = self.client.post(
            f'/sessions/{self.sid}/messages/{original["user_message_id"]}/variants/{second["assistant_message_id"]}/select',
        )
        self.assertEqual(forbidden.status_code, 409)
        self.assertEqual(db.visible_messages(self.sid), before)
        self.assertEqual(self.client.post(
            f'/sessions/{self.sid}/messages/{original["assistant_message_id"]}/regenerate',
        ).status_code, 409)

    def test_failed_regeneration_retains_previous_answer_and_rejects_late_completion(self):
        self.send("Question")
        original = self.finish("Saved answer")
        db.regenerate_response(self.sid, original["assistant_message_id"])
        stale = db.claim_generation(self.sid)
        db.interrupt_generations(self.sid, "Stopped", "cancelled")
        self.assertEqual(db.last_message(self.sid)["content"], "Saved answer")
        self.assertTrue(db.retry_generations(self.sid))
        fresh = db.claim_generation(self.sid)
        db.finish_generation(stale, "Late stale answer", "", None)
        self.assertEqual(db.response_variant_state(self.sid, original["user_message_id"])["count"], 1)
        db.finish_generation(fresh, "Fresh answer", "", None)
        self.assertEqual(db.response_variant_state(self.sid, original["user_message_id"])["count"], 2)
        self.assertEqual(db.last_message(self.sid)["content"], "Fresh answer")
        self.assertNotIn("Late stale answer", [m["content"] for m in db.list_messages(self.sid)])

    def test_regeneration_and_selection_reject_overlapping_and_foreign_requests(self):
        self.send("Question")
        original = self.finish("Answer")

        def regenerate(_):
            try:
                db.regenerate_response(self.sid, original["assistant_message_id"])
                return True
            except ValueError:
                return False

        with ThreadPoolExecutor(max_workers=6) as pool:
            accepted = list(pool.map(regenerate, range(6)))
        self.assertEqual(accepted.count(True), 1)
        self.assertEqual(self.client.post(
            f'/sessions/{self.sid}/messages/{original["user_message_id"]}/variants/{original["assistant_message_id"]}/select',
        ).status_code, 409)
        self.assertEqual(self.client.post(
            f'/sessions/{self.sid}/messages/{original["user_message_id"]}/fork',
        ).status_code, 409)
        db.interrupt_generations(self.sid, "Stopped", "cancelled")
        other = db.create_session(self.folder["id"])
        foreign = db.add_message(other["id"], "user", "Foreign question")
        for action in ("respond", "regenerate", "fork"):
            self.assertEqual(self.client.post(
                f'/sessions/{self.sid}/messages/{foreign["id"]}/{action}',
            ).status_code, 409)
        self.assertEqual([m["content"] for m in db.list_messages(other["id"])], ["Foreign question"])

    def test_deleting_selected_older_version_is_based_on_visible_order(self):
        self.send("Question")
        first = self.finish("First")
        db.regenerate_response(self.sid, first["assistant_message_id"])
        second = self.finish("Second")
        db.select_response_variant(self.sid, first["user_message_id"], first["assistant_message_id"])
        response = self.client.post(f'/sessions/{self.sid}/messages/{first["assistant_message_id"]}/delete')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(db.last_message(self.sid)["content"], "Second")
        self.assertEqual(db.response_variant_state(self.sid, first["user_message_id"])["ids"], [second["assistant_message_id"]])
        self.client.post(f'/sessions/{self.sid}/messages/{second["assistant_message_id"]}/delete')
        self.assertEqual(db.last_message(self.sid)["id"], first["user_message_id"])
        self.assertEqual(self.client.post(
            f'/sessions/{self.sid}/messages/{first["user_message_id"]}/respond',
        ).status_code, 200)

    def test_identical_uploads_share_data_across_requests_and_folder_deletion(self):
        files = [
            ("files", ("diagram.png", image_png(), "image/png")),
            ("files", ("paper.pdf", paper_pdf(), "application/pdf")),
        ]
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "First question"}, files=files)
        first_user = db.visible_messages(self.sid)[0]
        first = self.finish("First answer")
        other_folder = db.create_folder("Independent folder", "Different prompt")
        other = db.create_session(other_folder["id"], model="test-model")
        response = self.client.post(f'/sessions/{other["id"]}/send',
                                    data={"content": "Different question"}, files=files)
        self.assertEqual(response.status_code, 200, response.text)
        second_user = db.visible_messages(other["id"])[0]
        self.assertEqual(first_user["attachments_id"], second_user["attachments_id"])
        self.assertNotEqual(first_user["api_content_id"], second_user["api_content_id"])
        second_job = db.claim_generation(other["id"])
        db.finish_generation(second_job, "Second answer", "", None)
        urls = [part["image_url"]["url"] for part in first["messages"][-1]["content"]
                if part["type"] == "image_url"]
        self.assertEqual(len(urls), 3)
        for url in urls:
            # Even metadata, API content and different request snapshots share each image.
            self.assertEqual(db._conn.execute(
                "SELECT COUNT(*) FROM payloads WHERE instr(body, ?) > 0", (url,),
            ).fetchone()[0], 1)
        db.delete_folder(self.folder["id"])
        remaining = db.get_generation(other["id"], second_user["id"])
        prompt = self.client.get(
            f'/sessions/{other["id"]}/messages/{remaining["assistant_message_id"]}/prompt',
        )
        self.assertEqual(self.prompt_messages(prompt), second_job["messages"])
        self.assertEqual(db.visible_messages(other["id"])[0]["attachments"], first_user["attachments"])
        db.delete_folder(other_folder["id"])
        self.assertEqual(self.payload_stats(), (0, 0))
        self.assertEqual(db._conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_shared_search_is_copy_on_write_and_stale_attempts_leave_no_payloads(self):
        self.client.post(f"/sessions/{self.sid}/send",
                         data={"content": "Find this figure", "web_search": "true"},
                         files=[("files", ("figure.png", image_png(), "image/png"))])
        with patch.object(ollama, "web_search", side_effect=httpx.ConnectError("offline")), \
                patch.object(ollama, "chat_stream", side_effect=search_stream):
            generation._generate(db.claim_generation(self.sid))
        original = db.list_generations(self.sid)[0]
        self.assertEqual(original["trace"][-1]["status"], "failed")
        fork = db.fork_session(self.sid, original["user_message_id"])
        user = db.visible_messages(fork["id"])[0]
        shared = db.get_generation(fork["id"], user["id"])
        self.assertEqual(shared["trace"], [])
        self.assertEqual(shared["tools"], [])
        clean_messages = original["base_request"]["messages"]
        self.assertEqual(shared["messages"], clean_messages)
        # Decoded values cannot mutate shared persisted inputs or another branch.
        shared["messages"][-1]["content"][0]["text"] = "tampered"
        self.assertEqual(db.get_generation(fork["id"], user["id"])["messages"], clean_messages)
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["trace"], original["trace"])
        db.request_response(fork["id"], user["id"], web_search=True)
        results = [{"title": "Fresh source", "url": "https://example.org/fresh", "content": "New evidence"}]
        with patch.object(ollama, "web_search", return_value=results), \
                patch.object(ollama, "chat_stream", side_effect=search_stream):
            generation._generate(db.claim_generation(fork["id"]))
        updated = db.get_generation(fork["id"], user["id"])
        self.assertNotEqual(updated["request_id"], original["request_id"])
        self.assertEqual(json.loads(updated["trace"][-1]["message"]["content"])["results"], results)
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["trace"], original["trace"])
        self.assertEqual(db.visible_messages(fork["id"])[0]["attachments_id"],
                         db.visible_messages(self.sid)[0]["attachments_id"])
        db.retry_generations(self.sid)
        stale = db.claim_generation(self.sid)
        db.interrupt_generations(self.sid, "Stopped", "cancelled")
        before = self.payload_stats()
        self.assertFalse(db.record_tool_calls(stale, updated["trace"][0]["message"], "", None))
        self.assertFalse(db.start_tool(stale, "search-1"))
        self.assertFalse(db.finish_tool(stale, "search-1", {"results": results}))
        db.finish_generation(stale, "Too late", "", None)
        self.assertEqual(self.payload_stats(), before)
        db.delete_session(fork["id"])
        self.assertEqual(db.get_generation(self.sid, original["user_message_id"])["messages"], original["messages"])

    def test_fork_keeps_selected_prefix_snapshots_and_is_independent(self):
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Inspect diagram", "web_search": "true"},
                         files=[("files", ("diagram.png", image_png(), "image/png"))])
        results = [{"title": "Reference", "url": "https://example.org/paper", "content": "Value: 42"}]
        with patch.object(ollama, "web_search", return_value=results), \
                patch.object(ollama, "chat_stream", side_effect=search_stream):
            generation._generate(db.claim_generation(self.sid))
        original = db.list_generations(self.sid)[0]
        db.regenerate_response(self.sid, original["assistant_message_id"])
        self.finish("Unselected alternate")
        alternate = db.list_generations(self.sid)[0]
        self.assertEqual(alternate["trace"], [])
        self.assertEqual(alternate["messages"], original["base_request"]["messages"])
        self.assertEqual(db.get_message_generation(self.sid, original["assistant_message_id"])["trace"], original["trace"])
        db.select_response_variant(self.sid, original["user_message_id"], original["assistant_message_id"])
        self.send("Later excluded question")
        self.finish("Later excluded answer")
        db.update_session(self.sid, model="other-model", params_json='{"temperature":0.25}')
        before_rows, before_jobs = db.conversation_snapshot(self.sid)
        immutable_size = self.payload_stats()
        fork = db.fork_session(self.sid, original["assistant_message_id"])
        self.assertEqual(fork["folder_id"], self.folder["id"])
        self.assertEqual(fork["model"], "other-model")
        self.assertEqual(json.loads(fork["params_json"]), {"temperature": 0.25})
        cloned = db.visible_messages(fork["id"])
        self.assertEqual([m["content"] for m in cloned], ["Inspect diagram", "First answer"])
        self.assertTrue(set(m["id"] for m in cloned).isdisjoint(m["id"] for m in before_rows))
        self.assertEqual(cloned[0]["attachments"], before_rows[0]["attachments"])
        self.assertEqual(cloned[0]["api_content"], before_rows[0]["api_content"])
        self.assertEqual(cloned[0]["attachments_id"], before_rows[0]["attachments_id"])
        self.assertEqual(cloned[0]["api_content_id"], before_rows[0]["api_content_id"])
        self.assertEqual(self.payload_stats(), immutable_size)
        saved = db.get_message_generation(fork["id"], cloned[-1]["id"])
        self.assertEqual(saved["request_id"], original["request_id"])
        self.assertEqual(saved["messages"], original["messages"])
        self.assertEqual(saved["trace_id"], original["trace_id"])
        self.assertEqual(saved["trace"], original["trace"])
        self.assertEqual(saved["model"], "test-model")
        self.assertEqual(db.response_variant_state(fork["id"], cloned[0]["id"])["count"], 1)
        self.assertEqual(db.conversation_snapshot(self.sid), (before_rows, before_jobs))
        db.delete_session(self.sid)
        self.assertEqual([m["content"] for m in db.visible_messages(fork["id"])], ["Inspect diagram", "First answer"])
        page = self.client.get(f'/sessions/{fork["id"]}/messages/{cloned[-1]["id"]}/prompt')
        self.assertEqual(self.prompt_messages(page), original["messages"])
        db.delete_session(fork["id"])
        self.assertEqual(self.payload_stats(), (0, 0))
        self.assertEqual(db._conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_fork_at_user_waits_for_explicit_response_and_preserves_saved_request(self):
        self.send("Question")
        original = self.finish("Answer not included")
        self.start.reset_mock()
        response = self.client.post(f'/sessions/{self.sid}/messages/{original["user_message_id"]}/fork')
        self.assertEqual(response.status_code, 200, response.text)
        self.start.assert_not_called()
        fork = next(s for s in db.list_sessions(self.folder["id"]) if s["id"] != self.sid)
        cloned = db.visible_messages(fork["id"])
        self.assertEqual([m["content"] for m in cloned], ["Question"])
        self.assertFalse(any(j["status"] in ("queued", "running") for j in db.list_generations(fork["id"])))
        self.assertEqual(self.client.post(
            f'/sessions/{fork["id"]}/messages/{cloned[0]["id"]}/respond',
        ).status_code, 200)
        self.assertEqual(db.get_generation(fork["id"], cloned[0]["id"])["messages"], original["messages"])

    def test_translation_fork_uses_displayed_pair_order_not_raw_ids(self):
        db.set_folder_type(self.folder["id"], "translation")
        db.update_folder(self.folder["id"], "Translate", "Translate exactly", 100)
        self.send("a" * 240)
        first = self.finish("First translated chunk")
        self.finish("Second translated chunk")
        self.finish("Third translated chunk")
        fork = db.fork_session(self.sid, first["assistant_message_id"])
        self.assertEqual([m["role"] for m in db.visible_messages(fork["id"])], ["user", "assistant"])
        self.assertEqual(db.visible_messages(fork["id"])[0]["content"], "a" * 100)
        self.assertEqual(db.last_message(fork["id"])["content"], "First translated chunk")

    def test_legacy_regeneration_does_not_fabricate_original_prompt(self):
        db.add_message(self.sid, "user", "Legacy question")
        old = db.add_message(self.sid, "assistant", "Legacy answer")
        self.assertEqual(self.client.post(f'/sessions/{self.sid}/messages/{old["id"]}/regenerate').status_code, 200)
        newer = self.finish("New answer")
        legacy_page = self.client.get(f'/sessions/{self.sid}/messages/{old["id"]}/prompt')
        self.assertIn("best-effort reconstruction", legacy_page.text)
        self.assertEqual(self.prompt_messages(legacy_page)[-1]["content"], "Legacy question")
        new_page = self.client.get(f'/sessions/{self.sid}/messages/{newer["assistant_message_id"]}/prompt')
        self.assertEqual(self.prompt_messages(new_page), newer["messages"])

    def test_memo_fork_never_enqueues_a_model_request(self):
        db.set_folder_type(self.folder["id"], "memo")
        note = db.add_message(self.sid, "memo", "Private note")
        fork = db.fork_session(self.sid, note["id"])
        self.assertEqual([m["content"] for m in db.visible_messages(fork["id"])], ["Private note"])
        self.assertEqual(db.list_generations(fork["id"]), [])
        user = db.add_message(self.sid, "user", "Imported question")
        self.assertEqual(self.client.post(f'/sessions/{self.sid}/messages/{user["id"]}/respond').status_code, 409)
        self.assertEqual(db.list_generations(self.sid), [])
        self.start.assert_not_called()

    def test_deleting_response_updates_usage_without_recreating_it(self):
        self.send()
        job = self.finish(usage={"total_tokens": 12})
        response = self.client.post(f'/sessions/{self.sid}/messages/{job["assistant_message_id"]}/delete')
        self.assertEqual(response.status_code, 200)
        self.assertIn("Tokens: 0", response.text)
        self.assertNotIn("sse-connect", response.text)


class OllamaTests(unittest.TestCase):
    def test_malformed_or_unsafe_search_results_are_rejected(self):
        real_client = httpx.Client
        for data in ({}, {"results": [{"title": "Bad", "url": "javascript:alert(1)", "content": "x"}]},
                     {"results": [{"title": "Incomplete", "url": "https://example.org"}]},
                     {"results": [{"title": "Bad", "url": "https://exa\nmple.org", "content": "x"}]}):
            client = real_client(base_url="https://example.test", transport=httpx.MockTransport(
                lambda request, data=data: httpx.Response(200, json=data)
            ))
            with self.subTest(data=data), patch.object(ollama.httpx, "Client", return_value=client):
                with self.assertRaises(ValueError):
                    ollama.web_search("query")

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

    def test_streamed_tool_indexes_and_fragmented_arguments_are_preserved(self):
        chunks = self.stream([
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "call-", "type": "function",
                 "function": {"name": "web_", "arguments": '{"query":'}},
                {"index": 0, **search_call("first query", "first")},
            ]}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "second",
                 "function": {"name": "search", "arguments": '"second query","max_results":2}'}},
            ]}, "finish_reason": "tool_calls"}]}),
            'data: {"choices":[],"usage":{"total_tokens":19}}',
            'data: [DONE]',
        ])
        done = chunks[-1]
        second = search_call("second query", "call-second")
        second["function"]["arguments"] = '{"query":"second query","max_results":2}'
        self.assertEqual(done["tool_calls"], [search_call("first query", "first"), second])
        self.assertEqual(done["finish_reason"], "tool_calls")
        self.assertEqual(done["usage"]["total_tokens"], 19)

    def test_incomplete_tool_calls_never_become_completed_responses(self):
        broken = search_call()
        broken["function"]["arguments"] = '{"query":'
        for calls, finish in (([search_call()], "stop"), ([search_call()], "length"),
                              ([broken], "tool_calls"), ([], "tool_calls")):
            with self.subTest(calls=calls, finish=finish), self.assertRaises(ollama.ToolCallError):
                self.stream([
                    "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
                        {"index": index, **call} for index, call in enumerate(calls)
                    ]}, "finish_reason": finish}]}),
                    "data: [DONE]",
                ])

    def test_seed_is_sent_at_top_level_and_unsupported_options_are_omitted(self):
        payload = ollama._build_payload("test-model", [], {"seed": 42, "num_ctx": 8192, "top_k": 40})
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("options", payload)


class MigrationTests(unittest.TestCase):
    def test_inline_payload_migration_deduplicates_without_changing_history(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "copied.db"
            expected = legacy_payload_database(path)
            with patch.object(db, "DB_PATH", path), patch.object(db, "_conn", None):
                try:
                    db.init_db()
                    size = tuple(db._conn.execute("SELECT COUNT(*), SUM(LENGTH(body)) FROM payloads").fetchone())
                    db.init_db()
                    self.assertEqual(tuple(db._conn.execute(
                        "SELECT COUNT(*), SUM(LENGTH(body)) FROM payloads").fetchone()), size)
                    source = db.get_generation(1, 1)
                    fork = db.get_generation(2, 10)
                    self.assertEqual(source["request_id"], fork["request_id"])
                    self.assertEqual(source["messages"], expected["messages"])
                    self.assertTrue(all(event["origin"] == "legacy" for event in source["trace"]))
                    self.assertEqual(json.loads(source["trace"][-1]["message"]["content"])["results"],
                                     expected["search"]["results"])
                    clean = inputs.build_content("Figure question", db.list_messages(1)[0]["attachments"])
                    self.assertEqual(db.list_messages(1)[0]["api_content"], clean)
                    self.assertEqual(db.get_message_generation(1, 2)["messages"], expected["historical_messages"])
                    self.assertIsNone(db.get_message_generation(1, 4))
                    self.assertEqual(db.response_variant_state(1, 1)["selected_id"], 3)
                    self.assertEqual(db.get_generation(1, 20)["attempt"], 7)
                    self.assertEqual(db.get_generation(1, 20)["status"], "running")
                    self.assertEqual(db.get_session(1)["updated_at"], "2026-02-03")
                    self.assertEqual(db.list_messages(1)[0]["created_at"], "2026-01-01")
                    self.assertEqual(db._conn.execute(
                        "SELECT COUNT(*) FROM payloads WHERE instr(body, ?) > 0", (expected["url"],),
                    ).fetchone()[0], 1)
                    db.recover_generations()
                    self.assertEqual(db.get_generation(1, 20)["status"], "failed")
                    clean_fork = db.fork_session(1, 1)
                    clean_job = db.list_generations(clean_fork["id"])[0]
                    self.assertEqual(clean_job["messages"][-1], {"role": "user", "content": clean})
                    self.assertEqual(clean_job["trace"], [])
                    self.assertEqual(clean_job["tools"], [])
                    db.delete_session(clean_fork["id"])
                    db.delete_session(1)
                    self.assertEqual(db.get_message_generation(2, 11)["messages"], expected["messages"])
                    self.assertGreater(db.add_message(2, "memo", "New note")["id"], 5000)
                    self.assertEqual(db._conn.execute("PRAGMA foreign_key_check").fetchall(), [])
                    self.assertEqual(db._conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                finally:
                    db._conn.close()

    def test_payload_migration_failure_rolls_back_and_can_be_retried(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.db"
            expected = legacy_payload_database(path)
            with patch.object(db, "DB_PATH", path), patch.object(db, "_conn", None):
                real_store = payloads.store
                writes = 0

                def fail_midway(conn, value):
                    nonlocal writes
                    writes += 1
                    if writes == 3:
                        raise RuntimeError("Interrupted migration")
                    return real_store(conn, value)

                try:
                    with patch.object(payloads, "store", side_effect=fail_midway):
                        with self.assertRaisesRegex(RuntimeError, "Interrupted migration"):
                            db.init_db()
                    self.assertEqual(db._conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                    original = db._conn.execute("SELECT messages_json FROM generations WHERE user_message_id=1").fetchone()
                    self.assertEqual(json.loads(original[0]), expected["messages"])
                    self.assertEqual(db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 7)
                    db.init_db()
                    self.assertEqual(db.get_generation(1, 1)["messages"], expected["messages"])
                    self.assertEqual(db.get_generation(1, 1)["request_id"], db.get_generation(2, 10)["request_id"])
                    self.assertEqual(db._conn.execute("PRAGMA foreign_key_check").fetchall(), [])
                finally:
                    db._conn.close()

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
