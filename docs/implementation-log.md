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

## Step 2 — `backend/app/core/providers.py`

Added:
- `OllamaProviderError` (base), `OllamaUnreachableError`, `OllamaModelNotFoundError` — the two
  real boundary conditions when talking to Ollama (connection refused/timeout, and model not
  pulled). Anything else non-2xx is left to bubble up as a raw `httpx.HTTPStatusError` rather
  than being wrapped, per CLAUDE.md's "don't handle scenarios that can't happen."
- `GenerationResult` / `SpeedResult` frozen dataclasses. `GenerationResult.raw` keeps the full
  Ollama JSON as an escape hatch. Deliberately does NOT carry quantization or prompt-template
  version — those are Phase 2 run-loop/registry concerns, not this provider's job.
- `BaseLLMProvider(ABC)` with abstract `generate_response` and `test_speed`, matching the
  interface CLAUDE.md specifies. Used `ABC` over `Protocol` since the docs explicitly describe
  a future cloud provider *subclassing* this.
- `LocalOllamaProvider(BaseLLMProvider)` — POSTs to `{base_url}/api/generate` via
  `httpx.AsyncClient` with `stream: false` and the caller's `temperature` (default `0.0`, per
  CLAUDE.md's reproducibility rule for benchmark runs — callers doing ad-hoc/future work can
  still override it). `test_speed` is a thin wrapper over `generate_response` using a fixed
  warm-up prompt, computing tokens/sec from Ollama's `eval_count` / `eval_duration` (not
  `total_duration`, which would include one-time model load time and skew the number).
- Ephemeral `httpx.AsyncClient` per call rather than a persistent client on the instance — no
  lifespan hook exists yet to close a persistent one against, since the provider isn't wired
  into the FastAPI app in this step.

Adapted from a rough first draft found on stray branch `origin/main_om`, which used the wrong
class name (`OllamaProvider` instead of `BaseLLMProvider`/`LocalOllamaProvider`) and imported a
`config.settings` that branch never actually implemented.

## Step 3 — `backend/tests/test_providers.py`

Four cases: successful `generate_response`, successful `test_speed` (asserts an exact tokens/sec
figure to lock in the derivation formula), Ollama-unreachable, model-not-found.

No new dependency added. Deliberately avoided `pytest-asyncio` and `async def test_...` — the
repo's transitively-installed `anyio` would auto-register its pytest plugin, but its default
backend parametrization includes `trio`, which isn't installed and would fail those tests.
Instead, plain `def test_...()` functions drive the coroutines via `asyncio.run(...)`. Mocking
uses only `unittest.mock` (`AsyncMock`/`MagicMock` patching `httpx.AsyncClient.post`) — no
`respx` or similar added.
