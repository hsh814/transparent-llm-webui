"""File-based provider registrations; only environment-keyed providers are active."""

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    kind: str
    base_url: str
    api_key_env: str
    default_model: str = ""
    native_base_url: str | None = None


def _url(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"Invalid provider {field}; use an HTTP(S) URL without credentials.")
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in ("http", "https") and parsed.hostname
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment and "\\" not in value)
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid provider {field}; use an HTTP(S) URL without credentials.")
    return value.rstrip("/")


def _text(value: object, field: str, *, empty: bool = False) -> str:
    if (not isinstance(value, str) or value != value.strip() or (not empty and not value)
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ValueError(f"Invalid provider {field}.")
    return value


@lru_cache(maxsize=1)
def _registry(path: Path) -> tuple[Provider, ...]:
    try:
        registrations = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ValueError("Cannot read the provider configuration file. Check LLM_PROVIDERS_FILE or providers.json.") from None
    except (ValueError, UnicodeError):
        raise ValueError("The provider configuration file must contain a nonempty JSON array.") from None
    if not isinstance(registrations, list) or not registrations:
        raise ValueError("The provider configuration file must contain a nonempty JSON array.")
    result = []
    seen = set()
    fields = {"id", "name", "kind", "base_url", "api_key_env", "default_model", "native_base_url"}
    for registration in registrations:
        if not isinstance(registration, dict) or set(registration) - fields:
            raise ValueError("Invalid provider registration; only documented configuration fields are allowed.")
        identifier = registration.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identifier):
            raise ValueError("Invalid provider id; use 1–64 letters, digits, underscores, or hyphens.")
        if identifier in seen:
            raise ValueError("Provider ids must be unique.")
        seen.add(identifier)
        name = _text(registration.get("name"), "name")
        kind = registration.get("kind", "openai")
        if kind not in ("ollama", "openai"):
            raise ValueError("Invalid provider kind; use 'ollama' or 'openai'.")
        base_url = _url(registration.get("base_url"), "base_url")
        key_env = registration.get("api_key_env")
        if not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
            raise ValueError("Invalid provider api_key_env; specify an environment variable name.")
        default_model = _text(registration.get("default_model", ""), "default_model", empty=True)
        native_base = registration.get("native_base_url")
        if kind == "ollama":
            native_base = _url(native_base if native_base is not None else base_url.removesuffix("/v1"),
                               "native_base_url")
        elif native_base is not None:
            raise ValueError("native_base_url is supported only for Ollama providers.")
        result.append(Provider(identifier, name, kind, base_url, key_env, default_model, native_base))
    return tuple(result)


def list_providers() -> list[Provider]:
    """Return registrations with nonblank keys, in file order; keys stay server-side."""
    path = Path(os.environ.get("LLM_PROVIDERS_FILE", Path(__file__).with_name("providers.json"))).expanduser()
    available = []
    for provider in _registry(path):
        key = os.environ.get(provider.api_key_env, "")
        if not key.strip():
            continue
        if any(character.isspace() or not 33 <= ord(character) <= 126 for character in key):
            raise ValueError("A provider API key is invalid; check its api_key_env environment variable.")
        available.append(provider)
    return available


def get_provider(identifier: str) -> Provider:
    """Resolve exactly one configured provider; never substitute another provider."""
    for provider in list_providers():
        if provider.id == identifier:
            return provider
    raise ValueError("The selected provider is not configured or has no API key. Select an available provider.")


def default_provider() -> Provider:
    available = list_providers()
    if not available:
        raise ValueError("No providers have an API key. Set at least one api_key_env variable from providers.json in .env.")
    return available[0]
