# llm-litmus — Local LLM Evaluation Suite

Self-hosted tool that benchmarks local LLMs (via Ollama) against registered tests.
Full design doc: `docs/PROJECT_PLAN.md`. Read it before large changes.

## Architecture (non-negotiable constraints)

- **Local-only inference.** All model calls go to Ollama at `localhost:11434`. No cloud APIs, no API keys, anywhere. Cloud providers are a future phase.
- **Two-pass run loop.** Phase A: generate all target-model responses for a test, store raw responses in SQLite. Phase B: score stored responses. Never interleave generation and judge calls (forces Ollama model reloads).
- **Test Registry pattern.** Tests are declarative entries (dataset source, prompt template, scoring_type, scorer). The run loop, allocator, storage, and dashboard are type-blind — they iterate registered tests and call `scorer.score()`. Never branch on a specific test/dataset name outside the registry and scorers.
- **Two scorer types, one interface.** `BaseScorer.score(items) -> list[ScoreResult]`. `DeterministicScorer` = pure function, no model calls. `JudgeScorer` = batched calls to a user-selected local judge model, grounded rubric prompt, JSON output validated with retry-on-parse-failure. Only the scorer classes, the allocator (1x vs 2x call cost), and dashboard labels may be type-aware.
- **Provider abstraction.** All model access via `BaseLLMProvider` (`generate_response`, `test_speed`, `get_model_info`). `LocalOllamaProvider` is the only implementation for now. `get_model_info` reports quantization already baked into a pulled model tag (for reproducibility metadata) — the app never selects or applies quantization itself.

## Stack & conventions

- Backend: FastAPI + asyncio, Python 3.11+. Background work via FastAPI `BackgroundTasks` (no Celery/Redis).
- Frontend: Next.js (App Router) + Recharts. Talks to backend via HTTP polling (no WebSockets).
- Storage: stdlib `sqlite3`, no ORM. Tables: runs, responses, scores.
- Datasets: Hugging Face `datasets` with default local caching (no streaming mode). Public/ungated only — no HF token.
- No Docker. Entry point is `run.sh` (uv-managed root-level venv for backend, npm for frontend).
- Reproducibility: temperature 0 for all benchmark generation. Record model tag, quantization, prompt-template version, and decoding params with every run.
- Tests: pytest for allocator math, scorers, and provider interface. Keep scorers pure/testable without a live model.

## Repo layout

```
backend/app/
  core/         # providers.py, hardware.py, config
  database/     # sqlite3 helpers
  registry/     # test definitions + prompt templates
  scorers/      # deterministic/, judge/
  evaluations/  # allocator.py, two-pass run orchestration
  router/       # FastAPI endpoints
frontend/       # Next.js app + components
run.sh          # single entry point
```

## Style

- Prefer small pure functions; keep model I/O at the edges.
- No speculative abstractions beyond what's listed here (no provider factory until a second provider exists).
- Every new test should be addable as a registry entry without touching orchestration/UI. If a change breaks that property, stop and flag it.
