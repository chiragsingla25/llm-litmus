"""App-wide configuration lives here.

Defaults are loaded from config.json at the repo root; each field can still be overridden
via an environment variable of the same name for per-machine/CI overrides.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.json"


def _load_defaults() -> dict:
    with _CONFIG_PATH.open() as f:
        return json.load(f)


_defaults = _load_defaults()


@dataclass(frozen=True)
class Settings:
    OLLAMA_BASE_URL: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", _defaults["ollama_base_url"])
    )
    BACKEND_PORT: int = field(
        default_factory=lambda: int(os.environ.get("BACKEND_PORT", _defaults["backend_port"]))
    )
    FRONTEND_PORT: int = field(
        default_factory=lambda: int(os.environ.get("FRONTEND_PORT", _defaults["frontend_port"]))
    )
    DEFAULT_MODEL: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL", _defaults["default_model"])
    )
    DB_PATH: Path = field(
        default_factory=lambda: Path(
            os.environ.get("DB_PATH", str(_CONFIG_PATH.parent / _defaults["db_path"]))
        )
    )
    OLLAMA_REQUEST_TIMEOUT_SECONDS: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "OLLAMA_REQUEST_TIMEOUT_SECONDS", _defaults["ollama_request_timeout_seconds"]
            )
        )
    )
    OLLAMA_MAX_CONCURRENT_REQUESTS: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "OLLAMA_MAX_CONCURRENT_REQUESTS", _defaults["ollama_max_concurrent_requests"]
            )
        )
    )

    @property
    def cors_origin(self) -> str:
        return f"http://localhost:{self.FRONTEND_PORT}"


settings = Settings()
