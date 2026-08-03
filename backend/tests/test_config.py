import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from app.core.config import Settings

_CONFIG_JSON = json.loads((Path(__file__).resolve().parents[2] / "config.json").read_text())

_ENV_VARS = (
    "OLLAMA_BASE_URL",
    "BACKEND_PORT",
    "FRONTEND_PORT",
    "DEFAULT_MODEL",
    "DB_PATH",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS",
    "OLLAMA_MAX_CONCURRENT_REQUESTS",
)


def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults_come_from_config_json(monkeypatch):
    _clear_env(monkeypatch)
    settings = Settings()

    assert settings.OLLAMA_BASE_URL == _CONFIG_JSON["ollama_base_url"]
    assert settings.BACKEND_PORT == _CONFIG_JSON["backend_port"]
    assert settings.FRONTEND_PORT == _CONFIG_JSON["frontend_port"]
    assert settings.DEFAULT_MODEL == _CONFIG_JSON["default_model"]
    assert settings.DB_PATH == REPO_ROOT / _CONFIG_JSON["db_path"]
    assert settings.OLLAMA_REQUEST_TIMEOUT_SECONDS == _CONFIG_JSON["ollama_request_timeout_seconds"]
    assert settings.OLLAMA_MAX_CONCURRENT_REQUESTS == _CONFIG_JSON["ollama_max_concurrent_requests"]


def test_settings_env_vars_override_config_json(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1234")
    monkeypatch.setenv("BACKEND_PORT", "9001")
    monkeypatch.setenv("DEFAULT_MODEL", "some-other-model:latest")
    monkeypatch.setenv("DB_PATH", "/tmp/other.db")
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("OLLAMA_MAX_CONCURRENT_REQUESTS", "8")

    settings = Settings()

    assert settings.OLLAMA_BASE_URL == "http://example.com:1234"
    assert settings.BACKEND_PORT == 9001
    assert settings.DEFAULT_MODEL == "some-other-model:latest"
    assert settings.DB_PATH == Path("/tmp/other.db")
    assert settings.OLLAMA_REQUEST_TIMEOUT_SECONDS == 600.0
    assert settings.OLLAMA_MAX_CONCURRENT_REQUESTS == 8
    # Untouched field still falls back to config.json.
    assert settings.FRONTEND_PORT == _CONFIG_JSON["frontend_port"]


def test_cors_origin_is_derived_from_frontend_port(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FRONTEND_PORT", "4000")

    settings = Settings()

    assert settings.cors_origin == "http://localhost:4000"
