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

## Step 4 — verification

`ruff` isn't installed in the backend venv (it's configured in `pyproject.toml` but not listed
in `requirements.txt`, so presumably run via CI/a separate dev install) — substituted a manual
100-char line-length check (`pyproject.toml`'s configured limit) across the new/changed files;
all within limit. `pytest -q` in `backend/`: 5 passed (4 new provider tests + the existing
placeholder), no live Ollama instance required.

Real-Ollama sanity check: Ollama was found running locally with `llama3.2:latest` pulled. Ran
`LocalOllamaProvider.generate_response("llama3.2:latest", ...)` directly (not mocked) — got back
correct `text`, non-zero `prompt_tokens`/`completion_tokens`, and non-zero durations, confirming
the assumed Ollama response field names (`response`, `prompt_eval_count`, `eval_count`,
`eval_duration`, `total_duration`) are correct. `test_speed` on the same model returned
~27 tokens/sec, consistent with a small (3.2B, Q4) local model — confirms the
`eval_duration`-based derivation formula behaves sensibly end to end.

## Step 5 — migrate backend from pip+venv to uv (single root-level venv)

Prompted by the user hitting a failure running `pytest backend/tests`: VS Code (or an earlier
action) had run `uv init` at the repo root, creating a stray, near-empty uv project there
(`dependencies = []`, `requires-python = ">=3.12"`) with its own `.venv` containing only
`pytest`. `.vscode/settings.json` already pointed the editor's Python interpreter at that root
`.venv`, so any run through the editor (or a shell using that env) had no `fastapi`/`httpx`/etc.
installed — hence both the failed pytest run and the "Import httpx could not be resolved"
diagnostic on `providers.py`.

Decision (explicit user direction, not the default originally proposed): consolidate on a
**single root-level `.venv`**, matching the user's usual project layout, rather than scoping the
uv project to `backend/`. Concretely:
- Root `pyproject.toml` (previously the near-empty stray one) is now the real project file:
  `requires-python` corrected to `>=3.11` (was `>=3.12`, stricter than intended), dependencies
  set to the versions already verified working in the old `backend/.venv`
  (`fastapi>=0.139.2, uvicorn>=0.51.0, httpx>=0.28.1, datasets>=5.0.0`), dev group
  `pytest>=9.1.1` + `ruff` (ruff was previously undeclared — installed ad-hoc in CI only, never
  present in any local venv). `[tool.uv] package = false` made explicit — the `app` package
  under `backend/app` is never pip-installed, it resolves via cwd-relative imports, matching
  current behavior. `[tool.pytest.ini_options] testpaths` set to `["backend/tests"]`.
- Deleted: root `main.py` (unused `uv init` placeholder), root `.python-version` (no hard pin —
  keeps "any 3.11+, newest available" flexibility), `backend/pyproject.toml` and
  `backend/requirements.txt` (superseded), `backend/.venv/` (old pip venv) and the stray root
  `.venv` (regenerated clean via `uv sync`).
- `.vscode/settings.json` needed no change — it already pointed at
  `${workspaceFolder}/.venv/bin/python`, which is now correct.

This works without touching `backend/app`'s code because of two things confirmed true by
actually running it (not just assumed): pytest's default import mode walks up from
`backend/tests/__init__.py` to the first ancestor without an `__init__.py` (`backend/`) and
inserts that onto `sys.path`, so `app.core...` imports resolve regardless of where pytest itself
is invoked from — verified via `uv run pytest` from repo root, all 5 tests collected and passed.
uvicorn resolves `app.main:app` relative to its own process cwd (not the venv's location), so
`run.sh`'s existing `(cd backend && exec uvicorn app.main:app ...)` needs no change either.

Side effect of running `ruff` for the first time ever locally (it was never installed in any
venv before now): it caught 3 real pre-existing lint issues in `backend/tests/test_providers.py`
(unsorted imports, two `SIM117` nested-`with` cases) from the previous provider-implementation
step. Fixed those (one commit, separate from the uv migration itself) — auto-fixed the import
sort via `ruff check --fix`, manually combined the two nested `with` statements into single
multi-context statements. `ruff check backend` is now clean.

(Commit-boundary note: that lint-fix commit ended up also carrying the `backend/pyproject.toml`
and `backend/requirements.txt` deletions, since they were already `git rm`-staged from earlier
in this step and got swept in by an `git add` of just the test file. Content is correct either
way, just not the intended 1:1 step-to-commit mapping.)

## Step 6 — `run.sh`: uv-based backend setup

Removed the `python3.13/3.12/3.11` detection loop and its `PYTHON_BIN` variable — uv resolves
the interpreter itself against `requires-python` in `pyproject.toml`. Added a `uv` preflight
check in the same style as the existing `node`/`ollama` checks (clear error + install link,
`exit 1` if missing). Replaced the venv-creation + `pip install -r requirements.txt` +
requirements-hash-check block with `UV_PYTHON_DOWNLOADS=never uv sync --project "$ROOT_DIR"`
(the `never` flag preserves the old script's "fail loudly if no Python 3.11+ found" contract
rather than let uv silently download an interpreter — confirmed uv does exactly that when
allowed, since it had already cached one for the earlier stray root project), then
`source "$ROOT_DIR/.venv/bin/activate"`.

Kept the `hash_of` helper — it's still used by the (untouched) frontend section for
`package.json` change detection; only the backend's own requirements-hash usage was removable,
not the helper function itself. (Caught by review before committing — first draft of this edit
deleted `hash_of` entirely and would have broken frontend setup.)

Verified end-to-end: ran `./run.sh` in the background, confirmed `GET /health` returns
`{"status":"ok"}`, confirmed the frontend responds on `:3000`, confirmed via `ps aux` that the
running `uvicorn` process resolves to `/Users/chiragshome/projects/llm-litmus/.venv/bin/uvicorn`
(the new root venv, not the old `backend/.venv`), then stopped both processes.

## Step 7 — `.github/workflows/ci.yml`

Replaced both jobs' `actions/setup-python` + `pip install ...` steps with `astral-sh/setup-uv@v5`
+ `uv sync` + `uv run ruff check backend` / `uv run pytest`. No `working-directory` default
needed since `pyproject.toml` now lives at repo root, matching the checkout root. No
`actions/setup-python` step needed either — uv provisions its own interpreter in CI (a
disposable, network-connected runner, unlike a developer's machine, so this doesn't contradict
the `UV_PYTHON_DOWNLOADS=never` choice in `run.sh`). Not run against actual GitHub Actions in
this session (no push/PR yet) — will get exercised on the next push.

## Step 8 — `CLAUDE.md`

Updated the one line mentioning pip: "Entry point is `run.sh` (venv + pip for backend, npm for
frontend)" → "Entry point is `run.sh` (uv-managed root-level venv for backend, npm for
frontend)". Completes the backend uv migration — code, run.sh, CI, and docs are now all
consistent with the single root-level `.venv`.

## Step 9 — editable-install `app` as a real package; add `ipykernel`

Prompted by adding a `notebooks/` folder for manual experimentation: a notebook kernel started
from `notebooks/` has neither pytest's package-root-walking sys.path insertion nor uvicorn's
own-cwd insertion to fall back on, so `import app...` would fail there. First instinct was a
`sys.path.insert(...)` cell at the top of the notebook — user explicitly rejected that ("dont
use sys append path, it should be able to import script directly"). Correct fix: make `app` an
actually-installed (editable) package in the venv, not merely cwd-resolvable.

Added `[build-system]` (hatchling) + `[tool.hatch.build.targets.wheel] packages = ["backend/app"]`
to root `pyproject.toml`, and removed `[tool.uv] package = false` (which had explicitly opted
*out* of installing `app` — now the opposite of what's wanted). Added `ipykernel` to the `dev`
group for the VS Code Jupyter extension to use against the root `.venv` (confirmed with user:
VS Code's extension only needs `ipykernel`, not a full `jupyterlab`/`notebook` server).

`uv sync` built and editable-installed `llm-litmus==0.1.0` from `backend/app`. Verified:
`uv run python -c "import app; print(app.__file__)"` resolves to
`backend/app/__init__.py` both from repo root and from inside `notebooks/` with zero path
manipulation — confirms the fix is real, not just working by accident from one directory.
`uv run pytest` still 5/5 passing — no regression from the build-system change.

## Step 10 — `notebooks/provider_walkthrough.ipynb`

One notebook (per user's explicit preference over several small ones), five markdown-titled
sections run top to bottom: **Setup** (direct `from app.core... import ...`, no path
manipulation — relies on Step 9's editable install), **Check Ollama is reachable** (prints
`settings.OLLAMA_BASE_URL`, hits `/api/tags`, lists locally pulled models), **Generate a
response**, **Speed test**, and **Error handling** (both `OllamaUnreachableError` — bad base
URL — and `OllamaModelNotFoundError` — nonexistent model). Uses top-level `await` (ipykernel
supports this natively) rather than `asyncio.run(...)`, since that's the idiomatic notebook
style, not the plain-script style used in `backend/tests/test_providers.py`.

Committed with empty outputs — it's a template for the user to run themselves, not a snapshot of
one run. Before committing, extracted the exact cell code into an equivalent script and ran it
for real against the local Ollama instance (`llama3.2:latest`) to confirm every section actually
works: real generation output, ~16 tokens/sec speed reading, and both expected exceptions raised
with their intended messages. Confirmed `ipykernel`/Jupyter config resolves from the venv
(`uv run jupyter --paths` includes `.venv/etc/jupyter`), so VS Code's Jupyter extension should
be able to pick up the interpreter as a kernel directly.

No `notebooks/README.md` added — the notebook's own title/markdown cells cover it.

## Step 11 — centralize static config into root-level `config.json`

Prompted by the user noticing `config.py` hardcoded its one default inline, anticipating more
config growth, and pointing out other scripts already had their own static values that should
share one source: `backend/app/main.py` hardcoded the CORS origin `"http://localhost:3000"`
(silently duplicating `run.sh`'s `FRONTEND_PORT=3000`), `run.sh` hardcoded both ports, and the
notebook hardcoded `MODEL = "llama3.2:latest"`.

Scope confirmed with user: environment/deployment settings only (Ollama URL, ports, default
model) — not `providers.py`'s internal defaults (speed-test prompt, timeout, temperature), which
stay as function parameters since they're one class's implementation details, not shared config.
Ports explicitly included in the unification (not left as run.sh-only shell variables).

Added `config.json` at repo root (`ollama_base_url`, `backend_port`, `frontend_port`,
`default_model`). Rewrote `backend/app/core/config.py` to load defaults from it, keeping the
existing per-field env var override pattern (e.g. `OLLAMA_BASE_URL` env still wins if set — same
behavior as before, just JSON-backed instead of an inline literal). Added a derived
`cors_origin` property (`f"http://localhost:{FRONTEND_PORT}"`) instead of a separate stored
field — avoids reintroducing the exact kind of two-place duplication that motivated this change.
`config.py` itself is kept (not deleted) — it's the necessary single Python-side loader; deleting
it would just push JSON-parsing duplication into every consumer instead of removing it.

`Path(__file__).resolve().parents[3]` resolves `core/` → `app/` → `backend/` → repo root; this
depends on `app` being editable-installed (Step 9) so `__file__` always points at the real source
file regardless of caller cwd — verified by importing `settings` both from the repo root and
from inside `notebooks/`, both resolved all four fields correctly. `uv run pytest` still 5/5.

## Step 12 — `main.py`: derive CORS origin from config

`allow_origins=["http://localhost:3000"]` → `allow_origins=[settings.cors_origin]`. Verified by
importing the app object directly (`from app.main import app`) and inspecting
`app.user_middleware[0].kwargs["allow_origins"]` — resolves to `['http://localhost:3000']`,
same value as before but now derived from `FRONTEND_PORT` instead of duplicated as a literal.
