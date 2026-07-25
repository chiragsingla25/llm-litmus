import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "llm-litmus"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./litmus.db")

    class Config:
        env_file = ".env"


settings = Settings()
