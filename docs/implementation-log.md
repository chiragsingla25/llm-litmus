# Implementation Log

Running record of steps taken while implementing `BaseLLMProvider` + `LocalOllamaProvider`
(plan: see git history / PR description). Each entry corresponds to one commit, so any step
can be reverted independently via `git revert <sha>`.

## Step 1 — `backend/app/core/config.py`

Added a minimal `Settings` frozen dataclass with one field, `OLLAMA_BASE_URL`, defaulting to
`http://localhost:11434` and overridable via env var. Stdlib-only (`os.environ` + `dataclasses`)
— deliberately did not add `pydantic-settings` as a new dependency, since the project only
needs one setting right now and CLAUDE.md favors minimalism (no ORM, no speculative
abstractions).

Read once at import time as a module-level singleton (`settings = Settings()`), matching the
pattern the rest of the codebase will consume it with (`from app.core.config import settings`).
