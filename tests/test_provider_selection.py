"""Provider selection is a draft until a model completes the saved pair."""
import json
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("OLLAMA_API_KEY", "test-key")

from fastapi.testclient import TestClient
import db
import generation
import ollama
from app import app


class ProviderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(TemporaryDirectory())
        self.config = [
            {"id": "ollama", "name": "Ollama", "kind": "ollama", "base_url": "https://ollama.test/v1",
             "native_base_url": "https://ollama.test", "api_key_env": "FIRST_KEY", "default_model": "shared"},
            {"id": "second", "name": "Second", "base_url": "https://second.test/v1", "api_key_env": "SECOND_KEY"},
        ]
        config_path = Path(self.directory) / "providers.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.stack.enter_context(patch.dict(os.environ, {
            "LLM_PROVIDERS_FILE": str(config_path), "FIRST_KEY": "private-first", "SECOND_KEY": "private-second",
        }))
        self.stack.enter_context(patch.object(db, "DB_PATH", Path(self.directory) / "chat.db"))
        self.stack.enter_context(patch.object(db, "_conn", None))
        self.stack.enter_context(patch.object(ollama, "list_models", side_effect=lambda provider: {
            "ollama": ["shared", "first-only"], "second": ["shared", "second-only"],
        }[provider]))
        self.stack.enter_context(patch.object(ollama, "model_context_length", return_value=None))
        self.stack.enter_context(patch.object(generation, "start"))
        self.client = self.stack.enter_context(TestClient(app))
        self.stack.callback(lambda: db._conn.close())
        self.folder = db.create_folder("Providers", "Visible system")
        self.session = db.create_session(self.folder["id"])
        self.sid = self.session["id"]

    def pair(self):
        session = db.get_session(self.sid)
        return session["provider"], session["model"]

    def test_browsing_refreshing_and_abandoning_provider_preserves_saved_pair(self):
        response = self.client.get(f"/sessions/{self.sid}/models", params={"provider": "second"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('value="second-only"', response.text)
        self.assertNotIn('value="first-only"', response.text)
        self.assertEqual(self.pair(), ("ollama", "shared"))
        response = self.client.post(f"/sessions/{self.sid}/model/refresh", data={"provider": "second"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.pair(), ("ollama", "shared"))
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Use saved selection"})
        self.assertEqual(db.list_generations(self.sid)[0]["provider"], "ollama")
        response = self.client.get(f"/?session={self.sid}")
        self.assertIn('value="ollama" selected', response.text)
        self.assertNotIn("private-first", response.text)
        self.assertNotIn("private-second", response.text)

    def test_explicit_model_commits_pair_even_when_model_id_is_shared(self):
        response = self.client.post(f"/sessions/{self.sid}/model", data={"provider": "second", "model": "shared"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.pair(), ("second", "shared"))
        self.client.post(f"/sessions/{self.sid}/model", data={"temperature": "0.2"})
        self.assertEqual(self.pair(), ("second", "shared"))
        self.client.post(f"/sessions/{self.sid}/model", data={"provider": "", "model": ""})
        self.assertEqual(self.pair(), ("second", "shared"))
        self.client.post(f"/sessions/{self.sid}/send", data={"content": "Use second"})
        self.assertEqual(db.list_generations(self.sid)[0]["provider"], "second")

    def test_invalid_or_incomplete_pair_does_not_mutate_settings(self):
        for data in ({"provider": "second"}, {"provider": "missing", "model": "shared"},
                     {"provider": "second", "model": "first-only"}):
            with self.subTest(data=data):
                response = self.client.post(f"/sessions/{self.sid}/model", data=data)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(self.pair(), ("ollama", "shared"))
        with patch.object(ollama, "list_models", side_effect=RuntimeError("private-key")):
            response = self.client.post(f"/sessions/{self.sid}/model", data={"provider": "second", "model": "shared"})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("private-key", response.text)
            self.assertEqual(self.pair(), ("ollama", "shared"))
            self.client.post(f"/sessions/{self.sid}/model", data={"temperature": "0.3"})
            self.assertEqual(json.loads(db.get_session(self.sid)["params_json"])["temperature"], 0.3)

    def test_keyless_providers_are_hidden_and_saved_pair_is_not_reassigned(self):
        with patch.dict(os.environ, {"SECOND_KEY": ""}):
            response = self.client.get(f"/?session={self.sid}")
            self.assertNotIn('<select id="provider-choice"', response.text)
        with patch.dict(os.environ, {"FIRST_KEY": ""}):
            response = self.client.get(f"/?session={self.sid}")
            self.assertNotIn('<select id="provider-choice"', response.text)
            self.assertIn('id="provider-choice" type="hidden" name="provider" value="second"', response.text)
            self.assertIn('value="second-only"', response.text)
            self.assertEqual(self.pair(), ("ollama", "shared"))
            response = self.client.post(f"/sessions/{self.sid}/model", data={"temperature": "0.4"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.pair(), ("ollama", "shared"))
            response = self.client.post(f"/sessions/{self.sid}/send", data={"content": "Do not reroute"})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(db.list_messages(self.sid), [])
            response = self.client.get(f"/sessions/{self.sid}/models", params={"provider": "ollama"})
            self.assertEqual(response.status_code, 422)
            response = self.client.post(f"/sessions/{self.sid}/model", data={"provider": "second", "model": "shared"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.pair(), ("second", "shared"))

    def test_provider_without_default_requires_model_before_send(self):
        session = db.create_session(self.folder["id"], provider="second")
        self.assertEqual(session["model"], "")
        response = self.client.post(f'/sessions/{session["id"]}/send', data={"content": "No model"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(db.list_messages(session["id"]), [])
