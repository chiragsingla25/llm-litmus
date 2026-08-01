"""App-wide configuration lives here."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


settings = Settings()
