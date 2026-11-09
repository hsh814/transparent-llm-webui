"""Offline provider registration, routing, and cache isolation regressions."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

import ollama
import providers


def registration(identifier="alpha", **overrides):
    result = {"id": identifier, "name": identifier.title(),
              "base_url": f"https://{identifier}.example/v1/", "api_key_env": f"{identifier.upper()}_KEY"}
    result.update(overrides)
    return result


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config_path = Path(directory.name) / "providers.json"
        os.environ["LLM_PROVIDERS_FILE"] = str(self.config_path)
        ollama.clear_cache()
        self.addCleanup(ollama.clear_cache)

    def configure(self, *registrations):
        self.config_path.write_text(json.dumps(registrations), encoding="utf-8")
        providers._registry.cache_clear()
        for item in registrations:
            os.environ[item["api_key_env"]] = f"private-{item['id']}-key"

    def transport(self, handler):
        real_client = httpx.Client
        return patch.object(ollama.httpx, "Client", side_effect=lambda *args, **kwargs: real_client(
            *args, **kwargs, transport=httpx.MockTransport(handler)))

    def test_explicit_registration_is_authoritative_without_legacy_key(self):
        self.configure(registration(), registration("beta", default_model="same-model"))
        configured = providers.list_providers()
        self.assertEqual([item.id for item in configured], ["alpha", "beta"])
        self.assertEqual(configured[0].kind, "openai")
        self.assertEqual(configured[0].base_url, "https://alpha.example/v1")
        self.assertEqual(configured[0].default_model, "")
        self.assertIsNone(configured[0].native_base_url)
        self.assertEqual(providers.default_provider(), configured[0])
        self.assertEqual(providers.get_provider("beta").default_model, "same-model")
        self.assertNotIn("private-", repr(configured))
        os.environ["OLLAMA_API_KEY"] = "unused-legacy-key"
        with self.assertRaisesRegex(ValueError, "not configured"):
            providers.get_provider("ollama")
        self.assertEqual(len(providers.list_providers()), 2)

    def test_only_keyed_providers_are_available_and_default_skips_unkeyed(self):
        self.configure(registration(), registration("beta"), registration("gamma"))
        del os.environ["ALPHA_KEY"]
        os.environ["BETA_KEY"] = " \t"
        self.assertEqual([provider.id for provider in providers.list_providers()], ["gamma"])
        self.assertEqual(providers.default_provider().id, "gamma")
        with self.assertRaises(ValueError):
            providers.get_provider("alpha")
        os.environ["ALPHA_KEY"] = "enabled-key"
        self.assertEqual([provider.id for provider in providers.list_providers()], ["alpha", "gamma"])
        os.environ["ALPHA_KEY"] = ""
        del os.environ["GAMMA_KEY"]
        self.assertEqual(providers.list_providers(), [])
        with self.assertRaises(ValueError):
            providers.default_provider()

    def test_missing_file_does_not_fall_back_to_another_provider(self):
        os.environ["OLLAMA_API_KEY"] = "legacy-key"
        with self.assertRaises(ValueError):
            providers.default_provider()

    def test_configuration_changes_are_observed_lazily(self):
        self.configure(registration())
        self.assertEqual(providers.default_provider().id, "alpha")
        self.configure(registration("beta"))
        self.assertEqual(providers.default_provider().id, "beta")
        with self.assertRaises(ValueError):
            providers.get_provider("alpha")

    def test_invalid_registrations_fail_with_safe_messages(self):
        secret = "private-secret-credential"
        invalid = ["not JSON " + secret, "null", "{}", "[]", json.dumps([secret]),
                   json.dumps([registration(), registration()])]
        for override in ({"id": "bad/id"}, {"id": ""}, {"id": "x" * 65}, {"name": ""},
                         {"kind": "unknown"}, {"kind": []}, {"base_url": "ftp://example.org"},
                         {"base_url": f"https://user:{secret}@example.org/v1"},
                         {"base_url": f"https://example.org/v1?key={secret}"},
                         {"base_url": "https://example.org/#fragment"},
                         {"base_url": "https://example.org:invalid/v1"},
                         {"base_url": "https://example.org\n/v1"},
                         {"base_url": "https://example.org\\evil/v1"},
                         {"api_key_env": "BAD-ENV"}, {"api_key": secret},
                         {"default_model": None}, {"native_base_url": "https://native.example"}):
            invalid.append(json.dumps([registration(**override)]))
        os.environ["ALPHA_KEY"] = secret
        for config in invalid:
            with self.subTest(config=config):
                self.config_path.write_text(config, encoding="utf-8")
                providers._registry.cache_clear()
                with self.assertRaises(ValueError) as raised:
                    providers.list_providers()
                self.assertNotIn(secret, str(raised.exception))

    def test_all_credentials_validated_and_rotations_revalidated(self):
        self.configure(registration(), registration("beta"))
        self.assertEqual(len(providers.list_providers()), 2)
        for bad_key in ("secret with spaces", "secret\nheader", "non-ascii-é"):
            with self.subTest(key=bad_key):
                os.environ["BETA_KEY"] = bad_key
                with self.assertRaisesRegex(ValueError, "API key") as raised:
                    providers.get_provider("alpha")
                if bad_key:
                    self.assertNotIn(bad_key, str(raised.exception))
        del os.environ["BETA_KEY"]
        self.assertEqual(providers.default_provider().id, "alpha")

    def test_chat_requests_route_url_and_key_for_identical_model_ids(self):
        self.configure(registration(base_url="https://alpha.example/gateway/v1/"), registration("beta"))
        requests = []

        def respond(request):
            requests.append(request)
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "same-model")
            self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
            if payload["stream"]:
                return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"streamed"},'
                                      '"finish_reason":"stop"}]}\n\ndata: [DONE]\n')
            return httpx.Response(200, json={"choices": [{"message": {"content": "complete"}}]})

        with self.transport(respond):
            for provider in ("alpha", "beta"):
                messages = [{"role": "user", "content": "hello"}]
                self.assertEqual(ollama.chat_once("same-model", messages, {}, provider=provider)[0], "complete")
                chunks = list(ollama.chat_stream("same-model", messages, {}, provider=provider))
                self.assertEqual(chunks[0], {"type": "content", "text": "streamed"})
                self.assertEqual(chunks[-1]["type"], "done")
        self.assertEqual([str(request.url) for request in requests], [
            "https://alpha.example/gateway/v1/chat/completions"] * 2 + [
            "https://beta.example/v1/chat/completions"] * 2)
        self.assertEqual([request.headers["Authorization"] for request in requests], [
            "Bearer private-alpha-key"] * 2 + ["Bearer private-beta-key"] * 2)

    def test_model_cache_isolated_and_selected_refresh_retains_other_cache(self):
        self.configure(registration(), registration("beta"))
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"data": [{"id": "same-model"}, {"id": request.url.host}]})

        with self.transport(respond):
            alpha = ollama.list_models("alpha")
            beta = ollama.list_models(provider="beta")
            self.assertNotEqual(alpha, beta)
            self.assertEqual(ollama.list_models(provider="alpha"), alpha)
            self.assertEqual(ollama.list_models("beta"), beta)
            self.assertEqual(len(requests), 2)
            ollama.clear_cache("alpha")
            ollama.list_models("alpha")
            ollama.list_models("beta")
            self.assertEqual([request.url.host for request in requests], [
                "alpha.example", "beta.example", "alpha.example"])
            ollama.clear_cache()
            ollama.list_models("beta")
            self.assertEqual(len(requests), 4)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer private-alpha-key")
        self.assertEqual(requests[1].headers["Authorization"], "Bearer private-beta-key")

    def test_key_rotation_does_not_reuse_old_account_metadata(self):
        self.configure(registration())
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"data": [{"id": str(len(requests))}]})

        with self.transport(respond):
            self.assertEqual(ollama.list_models("alpha"), ["1"])
            os.environ["ALPHA_KEY"] = "rotated-key"
            self.assertEqual(ollama.list_models("alpha"), ["2"])
        self.assertEqual(requests[-1].headers["Authorization"], "Bearer rotated-key")

    def test_context_cache_isolated_and_generic_provider_has_no_native_request(self):
        self.configure(registration(kind="ollama"), registration("beta", kind="ollama",
                       native_base_url="https://beta-native.example/"), registration("generic"))
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(json.loads(request.content), {"model": "same-model"})
            value = 8192 if request.url.host == "alpha.example" else 32768
            return httpx.Response(200, json={"model_info": {"family.context_length": value}})

        with self.transport(respond):
            self.assertEqual(ollama.model_context_length("same-model", "alpha"), 8192)
            self.assertEqual(ollama.model_context_length(model="same-model", provider="beta"), 32768)
            self.assertEqual(ollama.model_context_length("same-model", provider="alpha"), 8192)
            self.assertIsNone(ollama.model_context_length("same-model", "generic"))
        self.assertEqual([str(request.url) for request in requests], [
            "https://alpha.example/api/show", "https://beta-native.example/api/show"])
        self.assertEqual([request.headers["Authorization"] for request in requests], [
            "Bearer private-alpha-key", "Bearer private-beta-key"])

    def test_native_tools_use_selected_ollama_provider_and_reject_generic(self):
        self.configure(registration(kind="ollama", native_base_url="https://native.example/"),
                       registration("generic"))
        requests = []

        def respond(request):
            requests.append(request)
            if request.url.path == "/api/web_search":
                self.assertEqual(json.loads(request.content), {"query": "query", "max_results": 2})
                return httpx.Response(200, json={"results": []})
            self.assertEqual(json.loads(request.content), {"url": "https://source.example/page"})
            return httpx.Response(200, json={"title": "page", "content": "text", "links": []})

        with self.transport(respond):
            self.assertEqual(ollama.web_search("query", 2, provider="alpha"), [])
            self.assertEqual(ollama.web_fetch("https://source.example/page", provider="alpha")["content"], "text")
            with self.assertRaisesRegex(ollama.ToolCallError, "not supported"):
                ollama.web_search("query", provider="generic")
            with self.assertRaisesRegex(ollama.ToolCallError, "not supported"):
                ollama.web_fetch("https://source.example/page", provider="generic")
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request.url.host == "native.example" for request in requests))
        self.assertTrue(all(request.headers["Authorization"] == "Bearer private-alpha-key" for request in requests))

    def test_outage_cache_does_not_poison_another_provider(self):
        self.configure(registration(), registration("beta"))
        requests = []

        def respond(request):
            requests.append(request)
            if request.url.host == "alpha.example":
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": "same-model"}]})

        with self.transport(respond):
            for _ in range(2):
                with self.assertRaises(httpx.HTTPStatusError):
                    ollama.list_models("alpha")
            self.assertEqual(ollama.list_models("beta"), ["same-model"])
        self.assertEqual(len(requests), 2)

    def test_unknown_provider_never_sends_request_or_exposes_input(self):
        self.configure(registration())
        with patch.object(ollama.httpx, "Client") as client:
            with self.assertRaises(ValueError) as raised:
                ollama.chat_once("same-model", [], {}, provider="secret-unknown-provider")
            self.assertNotIn("secret-unknown-provider", str(raised.exception))
            client.assert_not_called()

    def test_completion_errors_have_generic_safe_wording(self):
        self.configure(registration())
        for response in ('data: {"error":"private-upstream-error"}\n', 'data: invalid-json\n', ''):
            with self.subTest(response=response), self.transport(
                lambda request: httpx.Response(200, text=response)
            ):
                with self.assertRaises(RuntimeError) as raised:
                    list(ollama.chat_stream("same-model", [], {}, provider="alpha"))
                self.assertNotIn("Ollama", str(raised.exception))
                self.assertNotIn("private-upstream-error", str(raised.exception))
