"""App-wide configuration lives here.

Defaults are loaded from config.json at the repo root; each field can still be overridden
via an environment variable of the same name for per-machine/CI overrides.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.json"


def _load_defaults() -> dict:
    with _CONFIG_PATH.open() as f:
        return json.load(f)


_defaults = _load_defaults()


@dataclass(frozen=True)
class Settings:
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", _defaults["ollama_base_url"])
    BACKEND_PORT: int = int(os.environ.get("BACKEND_PORT", _defaults["backend_port"]))
    FRONTEND_PORT: int = int(os.environ.get("FRONTEND_PORT", _defaults["frontend_port"]))
    DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", _defaults["default_model"])

    @property
    def cors_origin(self) -> str:
        return f"http://localhost:{self.FRONTEND_PORT}"


settings = Settings()
