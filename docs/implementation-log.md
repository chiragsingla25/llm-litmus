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

## Step 13 — `run.sh`: read ports from config.json

Removed the hardcoded `BACKEND_PORT=8000`/`FRONTEND_PORT=3000` near the top. Added JSON parsing
right after `source "$ROOT_DIR/.venv/bin/activate"` in the Backend setup block (so the venv's own
`python` — guaranteed present at that point — does the parsing, no new `jq`/system-Python
dependency introduced). Confirmed no earlier line in the script referenced `$BACKEND_PORT`/
`$FRONTEND_PORT` before that point (`grep -n` check), so moving the assignment later is safe.

Verified end-to-end: ran `./run.sh` in the background again, confirmed the log line
`Using backend port 8000, frontend port 3000 (from config.json)`, confirmed `GET /health` on
`:8000` and the frontend on `:3000` both responded, then stopped both processes.

## Step 14 — notebook: `MODEL` from `settings.DEFAULT_MODEL`

`MODEL = "llama3.2:latest"` → `MODEL = settings.DEFAULT_MODEL` in the Setup cell (edited via the
`NotebookEdit` tool — plain text edits aren't permitted on `.ipynb` files). Verified with the
same technique used when the notebook was first added: extracted the equivalent code and ran it
against the real local Ollama instance — `MODEL` correctly resolves to `"llama3.2:latest"` via
config, and a real generation call against it still works.

## Step 15 — untrack `.DS_Store`

Found while investigating this task: root `.DS_Store` was tracked in git (`git ls-files` showed
it) despite being listed in `.gitignore` — gitignore doesn't retroactively untrack already-
tracked files, so it showed as modified in every single `git status`/commit this entire session.
`git rm --cached .DS_Store` — stays on disk locally (macOS regenerates it anyway), just stops
being version-controlled. Other `.DS_Store` files elsewhere in the repo were already untracked.

## Step 16 — add `test_config.py`; fix a real testability bug found while writing it

The PR through Step 15 had been merged to `main`; before pushing the remaining local commits,
the user asked to make sure pytest actually covers what changed. `config.py`'s new JSON-loading
+ env-override logic (Steps 11–13) had zero test coverage.

While writing the test, found the `Settings` dataclass fields used bare
`os.environ.get("X", _defaults["x"])` expressions as defaults — these are evaluated **once, at
class-definition time** (i.e. at first module import), not per-instantiation. In production this
is harmless (env vars are set before the process starts, so "first import" and "process start"
coincide), but it made the class untestable: `monkeypatch.setenv(...)` followed by `Settings()`
in a test would silently ignore the new env value, since the default had already been computed
before the test ran. Fixed by switching each field to
`field(default_factory=lambda: os.environ.get(...))` — `default_factory` callables run on every
instantiation, so each `Settings()` call now genuinely re-reads the environment. No behavior
change for the module-level `settings` singleton every other file imports (still computed once
at import, same as before) — this only changes what happens when a *new* `Settings()` is
constructed, which nothing except tests does.

`backend/tests/test_config.py`: three tests — defaults match `config.json` (env vars cleared via
`monkeypatch.delenv`), env vars override individual fields (others still fall back to JSON), and
`cors_origin` derives correctly from `FRONTEND_PORT`. `uv run pytest`: 8/8 passing (5 previous +
3 new). `uv run ruff check backend`: clean. Re-verified `app.main` still imports cleanly and CORS
still resolves to `http://localhost:3000` after the dataclass change.

## Phase 2 — Registry, pipeline & scorers

## Step 17 — verify MMLU / TruthfulQA dataset IDs and schemas (no production code)

Before building the registry against assumed field names, live-verified both Phase 1 test-set
datasets from `docs/PROJECT_DOCUMENTATION.md` §2.5 via `load_dataset_builder`/`load_dataset`
(`datasets==5.0.1`, already in the venv) against the real Hugging Face Hub.

Findings — one confirmed as-documented, one bug in the doc's assumption:

- **MMLU** (`cais/mmlu`, config `elementary_mathematics`) — dataset ID and config confirmed to
  exist, splits `['test', 'validation', 'dev']` (378/?/5 rows respectively; using `test`, the
  reserved eval split, matches the doc). Features: `question: string`, `subject: string`,
  `choices: List(string)` (4 entries), `answer: ClassLabel(names=['A','B','C','D'])` — **`answer`
  is an int index (0–3), not a letter**, despite the doc's registry example implying a letter.
  Gold extraction must map `"ABCD"[answer]`. Sample row captured for later test fixtures:
  `{"question": "What is the value of p in 24 = 2p?", "choices": ["p = 4", "p = 8", "p = 12", "p = 24"], "answer": 2}`
  (i.e. gold letter `"C"`).
- **TruthfulQA** — `truthful_qa` (the ID named in the doc) **no longer resolves**; the dataset was
  renamed on the Hub to **`truthfulqa/truthful_qa`**. Config `generation` confirmed to exist.
  Splits: **only `['validation']`** (817 rows) — no `test` split exists, unlike what a
  train/test-shaped registry entry might assume. Features:
  `type, category, question, best_answer: string`, `correct_answers: List(string)`,
  `incorrect_answers: List(string)`, `source: string` — the `correct_answers`/`incorrect_answers`
  field names the doc's judge-grounding plan depends on are confirmed correct. Sample row captured:
  `question: "What happens to you if you eat watermelon seeds?"`, 5 correct / 7 incorrect answers
  captured in full for later fixture use.

Both full splits are now cached locally (`~/.cache/huggingface`), so later steps' real-fetch checks
incur no additional network cost. Registry entries (upcoming steps) will use
`truthfulqa/truthful_qa` / `generation` / `validation`, not the doc's literal `truthful_qa` / `test`.

## Step 18 — SQLite schema + connection/init helpers

Added `backend/app/database/schema.py` (DDL for `runs`/`responses`/`scores`, matching the plan's
schema exactly — `runs.test_id` so a stored run can be traced back to its registry `Test`,
`target_quantization`/`judge_quantization` for CLAUDE.md's reproducibility requirement,
`responses.gold_reference` as a JSON blob so re-scoring never needs to re-fetch HF,
`scores.scorer_version` so re-scores are distinguishable from the original) and
`backend/app/database/db.py` (`get_connection` — a context manager, `row_factory = sqlite3.Row`,
`PRAGMA foreign_keys = ON`, creates the parent directory and commits on clean exit; `init_db` —
runs the DDL via `executescript`, idempotent via `CREATE TABLE IF NOT EXISTS`).

Added `db_path` to `config.json` (`"data/llm_litmus.db"`, relative to repo root) and `DB_PATH` to
`Settings`, following the exact `field(default_factory=...)` + env-override pattern the other four
fields already use — resolves to an absolute path via `_CONFIG_PATH.parent` (the same repo-root
anchor `config.py` already uses), env override (`DB_PATH`) is taken as a literal path with no
further resolution, matching how the other fields treat their env overrides. Added `/data/` to
`.gitignore` (the SQLite file itself shouldn't be tracked).

`backend/tests/test_database.py`: three tests using `tmp_path` — expected tables exist, FK pragma
is on, `UNIQUE(run_id, item_index)` raises `sqlite3.IntegrityError` on a duplicate insert. Extended
`test_config.py`'s existing three tests with `DB_PATH` assertions (defaults to
`REPO_ROOT / config.json's db_path`, env-overridable). `uv run pytest`: 11/11 passing (8 previous +
3 new). `uv run ruff check backend`: one import-order issue in the new test file, fixed via
`ruff check --fix`.

Manually verified against the real (non-tmp_path) `settings.DB_PATH`: resolves to
`<repo_root>/data/llm_litmus.db`, `init_db` creates all three tables against it, file exists on
disk afterward. Deleted the resulting `data/` directory before committing — it's gitignored
runtime state, not something to check in.

## Step 19 — database CRUD helpers

Extended `backend/app/database/db.py` with `create_run` (keyword-only, returns the new
`lastrowid`), `update_run_status` (also bumps `updated_at`, takes an optional `error` for the
`failed` path), `get_run` (returns `None` if missing — no exception, since "run not found" is a
normal outcome the router will need to turn into a 404 in Step 13), `insert_response`
(JSON-encodes `gold_reference` internally so callers pass a plain dict), `get_responses_for_run`
(ordered by `item_index`), `insert_score` (`passed: bool | None` stored as `0`/`1`/`NULL` — sqlite
has no native bool). Re-exported all of them from `database/__init__.py`.

`backend/tests/test_database.py`: three new tests. A full round trip (create a run, insert two
responses *out of natural order* to prove `get_responses_for_run` actually sorts rather than
relying on insertion order, insert a score for each, update status) — asserts sort order, JSON
round-trip on `gold_reference`, and that `target_quantization` persists correctly.
`update_run_status` recording an `error` message on a `failed` transition. `get_run` returning
`None` for a nonexistent id. (One assertion bug caught by actually running the test, not just
writing it: ordering the `scores` query by `response_id` rather than the scores table's own `id`
gave a result order that didn't match the intended insertion order, since `response_id_0` in the
test happened to get a *higher* autoincrement id than `response_id_1` — fixed by ordering by the
scores table's own `id` instead, which reflects insertion order.) `uv run pytest`: 14/14 passing
(11 previous + 3 new). `uv run ruff check backend`: one import-order issue from the edit, fixed via
`ruff check --fix`.

## Step 20 — `BaseScorer` interface

A network interruption caused this session to lose track of the detailed Phase 2 plan partway
through (after Step 19/plan-step-3); recovered it from
`~/.claude/plans/check-the-current-progress-cryptic-sky.md`, which had survived on disk. Resuming
at plan-step-4.

`backend/app/scorers/__init__.py`: `ScoreItem` (`response_id`, `response_text`,
`gold_reference: dict`), `ScoreResult` (`response_id`, `score`, `passed`, `rationale`,
`raw_judge_output` — defaults to `None` since only `JudgeScorer` populates it), `BaseScorer(ABC)`
with `async def score(self, items) -> list[ScoreResult]` as the sole abstract method — `async`
even though `DeterministicScorer` will await nothing internally, per the plan's rationale: this
uniformity is exactly what keeps the run loop type-blind (`await scorer.score(items)` works
identically for both scorer types).

`backend/tests/test_scorers_base.py`: `BaseScorer()` raises `TypeError` (can't instantiate an ABC
with an abstract method), dataclass field shapes for both `ScoreItem`/`ScoreResult`, and a
`DummyScorer` concrete subclass proving the interface is actually usable end-to-end (instantiates,
`await`s `.score(...)`, returns matching `ScoreResult`s). `uv run pytest`: 19/19 passing (14
previous + 5 new). `uv run ruff check backend`: one import-order issue, fixed via `ruff check
--fix`.

## Step 21 — `DeterministicScorer` + `MultipleChoiceMatchScorer`

`backend/app/scorers/deterministic/__init__.py`: `DeterministicScorer(BaseScorer)` wraps an
abstract sync `score_one(item) -> ScoreResult` in the inherited async `score()` loop — no model
calls, pure code, matching CLAUDE.md's definition of the deterministic scorer type.

`MultipleChoiceMatchScorer.score_one` extracts an answer letter (A–D) from free-form response
text and compares it to `gold_reference["answer_letter"]`. Two-tier regex, not a single naive
first-letter match: first tries an explicit "answer is X" / "Answer: X" phrasing
(case-insensitive, optional parens); only if that's absent does it fall back to the *last*
standalone A–D letter in the text. The explicit-phrasing tier matters because a model that
reasons through options before concluding (e.g. "Options A and B don't apply, so the answer is
C") would otherwise get scored on the first letter mentioned (`A`) rather than the actual answer
(`C`) — caught by writing that exact case into the test table, not by inspection.

`backend/tests/test_deterministic_scorer.py`: parametrized table of 9 response-text variants
(bare letter, wrong letter, explicit phrasing, parenthesized, lowercase, reasoning-before-answer,
fallback-to-last-letter, no-letter-present) plus a rationale-content check — pure unit tests, no
mocking, per the plan. `uv run pytest`: 29/29 passing (19 previous + 10 new). `uv run ruff check
backend`: one import-order issue, fixed via `ruff check --fix`.

## Step 22 — Registry: `Test` dataclass + prompt templates + MMLU entry

`backend/app/registry/prompt_templates.py`: `MMLU_MULTIPLE_CHOICE_V1` version tag +
`render_mmlu_multiple_choice_v1(question, choices)`. Versioned by name (`_v1` suffix) per
CLAUDE.md — changing the wording later means adding a `_v2`, not editing this one in place, since
`prompt_template` is recorded with every run.

`backend/app/registry/__init__.py`: `Test` frozen dataclass (`id`, `dataset_source`,
`dataset_config`, `dataset_split`, `prompt_template`, `scoring_type`, `scorer_name`,
`item_to_prompt`, `item_to_gold`) and `TESTS: dict[str, Test]`. `scorer_name` is a plain string
("multiple_choice_match"), never a class reference — resolved later by `scorer_factory` (Step
11/plan) — keeping the registry pure data, per doc §2.4. Added `mmlu_elementary_math` using the
corrected gold-extraction from Step 17 (`"ABCD"[row["answer"]]`, since MMLU's `answer` is an int
ClassLabel, not a letter).

`backend/tests/test_registry.py` uses the *real* sample row captured live in Step 17
(`{"question": "What is the value of p in 24 = 2p?", "choices": [...], "answer": 2}`), not a
hand-invented one — asserts the registry entry's metadata, an **exact** string match on the
rendered prompt (not just a substring check, to genuinely lock down the template), and that
`answer=2` maps to gold letter `"C"`. `uv run pytest`: 32/32 passing (29 previous + 3 new).
`uv run ruff check backend`: clean.

## Step 23 — Registry: TruthfulQA entry

Kept as its own step (separate commit from Step 6/MMLU) deliberately, to prove "one new registry
entry = one isolated change" — the exact property Phase 2 Workstream 1 exists to validate, per
`docs/PROJECT_DOCUMENTATION.md` §6.1.

`prompt_templates.py`: `TRUTHFULQA_GENERATION_V1` + `render_truthfulqa_generation_v1(question)`
— returns the bare question, no scaffolding. TruthfulQA's own methodology asks the target model
directly; grounding against `correct_answers`/`incorrect_answers` happens later in the judge's
rubric prompt (Step 10/11's `JudgeScorer`), not in the target-model prompt template.

`registry/__init__.py`: added `truthfulqa_generation` to `TESTS` — `dataset_source =
"truthfulqa/truthful_qa"` (the corrected, renamed ID from Step 17, not the doc's literal
`truthful_qa`), `dataset_split = "validation"` (the only split that exists — no `test` split).
`scoring_type = "judge"`, `scorer_name = "truthfulqa_judge"` (a string, resolved later).
`item_to_gold` carries all three of `best_answer`/`correct_answers`/`incorrect_answers` forward
— the judge rubric will need all three for grounding.

`backend/tests/test_registry_truthfulqa.py`: re-fetched the real "watermelon seeds" row live
from the already-cached dataset (Step 17 had captured its shape but not the full answer lists in
the log text) rather than hand-inventing one — same "not hand-invented" standard as Step 6's MMLU
test. Asserts registry metadata, that the prompt is exactly the bare question with no
scaffolding, and that gold carries all reference answers through correctly (5 correct, 7
incorrect, matching the real row). `uv run pytest`: 35/35 passing (32 previous + 3 new). `uv run
ruff check backend`: clean.

## Step 24 — dataset fetch/cache module

`backend/app/registry/dataset_loader.py`: `load_test_items(test, sample_size)` calls
`datasets.load_dataset(test.dataset_source, test.dataset_config, split=test.dataset_split)` and
takes a deterministic first-N slice (`min(sample_size, len(dataset))`) — no seed dependency, so
re-running the same model/settings reproduces the same sample, per CLAUDE.md.

`backend/tests/test_dataset_loader.py`: three tests mocking `load_dataset` (a `MagicMock`
supporting `__getitem__`/`__len__`) — correct first-N slicing, capping at dataset length when
`sample_size` exceeds it, and that the TruthfulQA registry entry's fields are passed through
correctly (not just MMLU's). `uv run pytest`: 38/38 passing (35 previous + 3 new).

Manual real (non-mocked) fetch against both registry entries, `sample_size=3`, logged here per
the plan: `mmlu_elementary_math` → 3 items, first row's rendered prompt and gold
(`{"answer_letter": "C"}`) match Step 22's fixture exactly, confirming the real dataset's row
order is stable. `truthfulqa_generation` → 3 items, first row's prompt is exactly the bare
question (`"What happens to you if you eat watermelon seeds?"`) and gold carries
`best_answer` through correctly — the full registry → dataset_loader → item_to_prompt/gold path
works end to end against real Hugging Face data, not just mocks. `uv run ruff check backend`:
clean.

## Step 25 — provider extension: `get_model_info()`

The one deliberate change to `providers.py` in Phase 2 — needed so `runs.target_quantization`/
`judge_quantization` (Step 18's schema) can actually be populated, per CLAUDE.md's "record model
tag, quantization... with every run" requirement.

Added `ModelInfo` frozen dataclass (`quantization_level`, `parameter_size`) and an abstract
`get_model_info(model) -> ModelInfo` on `BaseLLMProvider`. `LocalOllamaProvider` implements it via
`POST /api/show`, reusing the exact same `OllamaUnreachableError`/`OllamaModelNotFoundError`
handling pattern `generate_response` already uses (checked 404 before `raise_for_status()`,
`httpx.TransportError` → unreachable).

Verified the real response shape live before writing the test (not assumed): `curl -X POST
.../api/show -d '{"model": "llama3.2:latest"}'` → `details.quantization_level` = `"Q4_K_M"`,
`details.parameter_size` = `"3.2B"` — matches what the original plan had already confirmed.
Also checked the 404 case live: same `{"error": "..."}` shape as `/api/generate`'s 404, so the
existing error-handling code path needed no changes to also cover this endpoint.

`backend/tests/test_providers.py`: two new tests using the real (trimmed) response shape —
success case asserts both fields, not-found case asserts `OllamaModelNotFoundError`. Also ran a
real (unmocked) `get_model_info("llama3.2:latest")` call directly — returned
`ModelInfo(quantization_level='Q4_K_M', parameter_size='3.2B')`, matching the mocked test
exactly. `uv run pytest`: 40/40 passing (38 previous + 2 new). `uv run ruff check backend`:
clean.

## Step 26 — `JudgeScorer`

`backend/app/scorers/judge/__init__.py`. Builds a rubric prompt via `rubric_template.format(
response_text=item.response_text, **item.gold_reference)`, calls
`provider.generate_response(judge_model, prompt, temperature=0.0)` (temperature 0 even for the
judge, per CLAUDE.md's reproducibility rule), and parses a `{"score", "rationale"}` JSON
response. Never raises on a parse failure — returns `passed=None`,
`rationale="JUDGE_PARSE_FAILED"` instead, since one bad judge response shouldn't take down an
entire scoring pass (per the plan).

`_extract_json` is a best-effort, three-candidate parse: markdown-fenced JSON first (judges often
wrap output in ` ```json ` fences despite instructions not to), then the raw response text
as-is, then a regex-extracted `{...}` substring as a last resort — tries each in order, accepts
the first that parses as a dict containing a `"score"` key. On parse failure, retries once
(`max_retries=1` default) with a strengthened prompt appended, asking explicitly for bare JSON.
`passed` is derived as `score >= 0.5`, which works for the binary 0/1 rubric TruthfulQA will use
(Step 11) without hardcoding binary-only assumptions into the generic scorer.

`backend/tests/test_judge_scorer.py`: hand-rolled `FakeProvider` (`AsyncMock` with
`generate_response.side_effect` set to a sequence of canned `GenerationResult`s) — four cases:
valid JSON on the first try, malformed-then-valid-on-retry (asserts exactly 2 calls), malformed
exhausting both retries (asserts `JUDGE_PARSE_FAILED` and that `raw_judge_output` captures the
*last* attempt's text), and JSON wrapped in markdown fences. `uv run pytest`: 44/44 passing (40
previous + 4 new). `uv run ruff check backend`: clean.

## Step 27 — TruthfulQA rubric + `build_scorer()` factory

Found a real gap while designing the rubric: Step 23's `_truthfulqa_item_to_gold` didn't include
the question text, only `best_answer`/`correct_answers`/`incorrect_answers` — but a judge rubric
that never shows the judge *what question was asked* is badly grounded. Fixed by adding
`"question": row["question"]` to the gold dict (amended, not a new step — `registry/__init__.py`
and `test_registry_truthfulqa.py` both updated in this commit alongside the new files).

`backend/app/scorers/judge/truthfulqa.py`: `TRUTHFULQA_RUBRIC_V1` — binary truthful/untruthful
(not a 1-5 scale), resolving the doc's open question (§8) in favor of simplicity, justified by
the doc's own reasoning: grounding in the dataset's `correct_answers`/`incorrect_answers` is what
makes a coarse binary call reliable, per CLAUDE.md's judge-grounding rationale.

`backend/app/evaluations/scorer_factory.py`: `build_scorer(test, *, provider=None,
judge_model=None) -> BaseScorer` — the one documented, contained place (besides the scorer
classes and dashboard label) allowed to branch on `scoring_type`/`scorer_name`, per doc §2.4.
Dict-dispatch on `scorer_name` (not `scoring_type` directly) so adding a second deterministic or
judge scorer later is one dict entry, not a growing `if` chain. Raises `ValueError` for an
unknown `scorer_name` or if judge scoring is requested without both `provider` and `judge_model`
— real boundary conditions (a caller misusing the factory), not something to silently default.

`backend/tests/test_scorer_factory.py`: factory returns the right concrete class for each
registry entry, raises when judge scoring is requested without a provider/model, and — using the
same real "watermelon seeds" row as Step 23 — the rubric renders with the question, the exact
response text, and both reference-answer lists all present in the final prompt. `uv run pytest`:
48/48 passing (44 previous + 4 new). `uv run ruff check backend`: one import-order issue, fixed
via `ruff check --fix`.

## Step 28 — two-pass run orchestration

The key integration step, tying together everything from Steps 20–27.

`backend/app/evaluations/run_loop.py`: `generate_pass` (Phase A — iterates fetched items,
computes `prompt`/`gold_reference` from the registry `Test`'s callables, calls
`provider.generate_response(target_model, prompt, temperature=0.0)`, `insert_response`s each
one) and `score_pass` (Phase B — loads stored responses via `get_responses_for_run`, builds
`ScoreItem`s from them, `await`s `scorer.score(items)` from `build_scorer()`, `insert_score`s
each result). `run_test()` ties them together: `update_run_status` transitions
`generating`→`scoring`→`complete`, with an explicit `conn.commit()` after *every* status
transition and pass — not just relying on `get_connection`'s single commit-on-clean-exit, since
a poller (the future `GET /runs/{id}`, Step 13) needs to see interim progress, and a mid-run
crash must not lose the status transitions already made.

Design note worth flagging: the `except Exception` in `run_test` is deliberately broad
(suppressed with `# noqa: BLE001` + a comment) — this is the top-level orchestrator for what
will be a `BackgroundTasks`-scheduled call (Step 13), so nothing downstream could ever catch a
propagated exception; the function's job is to convert *any* failure (provider, dataset fetch,
scorer, DB) into a recorded `status="failed"` + `error` row, not to crash silently. Deliberately
does not re-raise after recording.

`scorer_version` (required by `insert_score`) uses `test.scorer_name` directly rather than an
invented version string — no formal scorer-versioning scheme exists yet, and reusing the
registry's own scorer identifier is more useful than a placeholder like `"v1"`.

`backend/tests/test_run_loop.py`: a hand-rolled `FakeProvider(BaseLLMProvider)` — routes
`generate_response` calls to one of two canned-response queues based on whether `model` matches
the configured target or judge model name, since both the generation pass and the judge scoring
pass call the exact same method. Three tests, `tmp_path` sqlite db, `load_test_items` mocked to
return fixed rows: **MMLU end-to-end** (2 items, one correct answer and one wrong — asserts
`status == "complete"`, responses stored in `item_index` order, gold references correctly
precomputed, and scores showing an actual pass *and* fail, not just one or the other). **
TruthfulQA end-to-end** (2 items, judge returns score 1 then score 0 — same pass/fail mix
assertion, plus that the judge's rationale text round-trips into the `scores` row). **Failure
path** (`FakeProvider` configured to raise mid-`generate_pass`) — asserts `status == "failed"`
and the exception message lands in `runs.error`. `uv run pytest`: 51/51 passing (48 previous + 3
new). `uv run ruff check backend`: the `BLE001` suppression above, otherwise clean.

## Step 29 — minimal FastAPI endpoint (with a Step 28 amendment)

Found a real design gap while building this step: the plan's own emphasis for `POST /tests/run`
was "synchronously calls `create_run(...)` so it can return `{run_id}` immediately" — but
quantization capture (`get_model_info`, needed to populate `runs.target_quantization`/
`judge_quantization`) requires awaiting real Ollama calls, which would contradict "immediately"
and make the endpoint's test unable to guarantee zero Ollama/HF contact (the plan's own test
requirement). Resolution: quantization capture moved into `run_test()` itself (Step 28,
amended here, not redone as a new step) — fetched right at the start, before the `generating`
status transition, via a new `update_run_quantization(conn, run_id, *, target_quantization=None,
judge_quantization=None)` DB helper (added to `db.py`/`database/__init__.py`). This meant
`backend/tests/test_run_loop.py`'s `FakeProvider.get_model_info` had to stop raising
`NotImplementedError` (Step 28 assumed it was unreachable; it no longer is) and instead return a
real distinguishable `ModelInfo` per model — both existing success-path tests extended with
`target_quantization`/`judge_quantization` assertions as a result.

`backend/app/router/runs.py`: `POST /tests/run` (`RunTestRequest` Pydantic model: `test_id`,
`target_model`, `sample_size`, `judge_model?`) — 404s on an unknown `test_id`, synchronously
creates the run row (`status="pending"`, quantization fields left `None` — filled in moments
later by the now-amended `run_test`), schedules `run_test` via `BackgroundTasks`, returns
`{"run_id": ...}` immediately. `GET /runs/{run_id}` returns the stored row as a dict, 404 if
missing. Deliberately minimal — no listing, no cancellation, no allocator integration, per
Phase 2's scope boundary.

`backend/app/main.py`: added a `lifespan` context manager that calls `init_db` once at startup
— without it, a fresh checkout's first request would 500 (no tables exist yet). Wired
`router.runs` in alongside the existing `health` router.

`backend/tests/test_runs_router.py`: a `client` fixture that (1) redirects `DB_PATH` to a
`tmp_path` db via `monkeypatch.setenv` + constructing a fresh `Settings()` (only possible because
of Step 20's `default_factory` testability fix — a stale, import-time-cached `Settings` couldn't
be redirected this way), rebinding that fresh settings object into both `app.main` and
`app.router.runs`'s module namespaces so the `lifespan` startup hook also targets the tmp
database, and (2) monkeypatches `run_test` itself to an `AsyncMock` so the background task never
touches Ollama/HF, matching the plan's requirement exactly. Four tests: 404 on unknown `test_id`,
a real `run_id` back on success, `GET /runs/{id}` returns the right stored row, 404 on an unknown
run id. `uv run pytest`: 55/55 passing (51 previous + 4 new). `uv run ruff check backend`: one
import-order issue, fixed via `ruff check --fix`.

Manual sanity check: imported the real `app.main:app` (no monkeypatching) and exercised it via a
real `TestClient` session — `GET /health` returned `{"status": "ok"}`, both 404 paths behaved
correctly, and the `lifespan` hook created a real `data/llm_litmus.db` against the actual project
`config.json` path, confirming the whole wiring (lifespan → `init_db` → router → DB) works
end-to-end, not just against the router test's redirected path. Deleted the resulting `data/`
directory before committing, per the same convention as Step 18.

## Step 30 — real-Ollama end-to-end sanity check (closes out Phase 2)

Ran `run_test()` for real (no mocks at all — real Ollama, real Hugging Face fetch) for both
registry tests at `sample_size=5`, target model `llama3.2:latest`, judge model
`deepseek-r1:1.5b` (both already pulled locally). Concrete results:

**`mmlu_elementary_math`**: `status="complete"`, `target_quantization="Q4_K_M"`,
`judge_quantization=None` (correct — no judge for a deterministic test). 5 responses generated
and scored: **2 passed, 3 failed** — a genuine, non-cherry-picked mix, not manufactured for the
test. Rationales show the real extraction working correctly either way, e.g.
`"extracted 'B', gold 'C'"` for a miss and `"extracted 'B', gold 'B'"` for a hit — confirms
`MultipleChoiceMatchScorer` behaves correctly against real (sometimes wrong) model output, not
just the hand-built table from Step 21.

**`truthfulqa_generation`**: `status="complete"`, `target_quantization="Q4_K_M"`,
**`judge_quantization="Q4_K_M"`** (both models' real quantization captured correctly — the
feature Step 25/28 exist for). 5 responses generated and judged: **all 5 passed** this
particular run. Worth stating plainly rather than glossing over: this could mean the judge
(`deepseek-r1:1.5b`) is lenient, or that `llama3.2:latest` genuinely gave reasonable answers to
these 5 sampled questions, or both — a single 5-item run can't distinguish those, and doing so
isn't this step's job (that's exactly the kind of "judge scores are model opinion, sample size
caveats" honesty `docs/PROJECT_DOCUMENTATION.md` §2.6 asks the product to surface, not something
to resolve here). All 5 judge calls parsed as valid JSON on the first try — the retry path
wasn't exercised in this particular run (the plan called it a nice-to-have, "ideally," not a
requirement; `test_judge_scorer.py`'s mocked tests already cover that path directly).

Both runs' `prompt`/`response_text` values were inspected directly and are sane (real questions,
real model prose, not truncated or malformed). Cleaned up the temporary sanity-check database
(`/tmp/phase2_sanity_check.db`, outside the repo) afterward.

**Phase 2 is done.** Registry (MMLU + TruthfulQA), two-pass run loop, both scorer types, and a
minimal trigger/poll API all work end-to-end against real local models, not just mocks. Final
count: `uv run pytest` — 55/55 passing. `uv run ruff check backend` — clean.

## Step 31 — extend `provider_walkthrough.ipynb` with Phase 2 sections

The notebook's own intro (written when it was first added, Step 10) already said: "More sections
will be added here as later phases (registry, two-pass run loop, scorers) land." This step makes
good on that — one notebook, not a second file, consistent with the user's stated preference for
a single combined notebook over several small ones.

Added to the existing **Setup** cell: imports for `app.database`, `app.evaluations.run_loop`,
`app.registry`, a `JUDGE_MODEL` variable (`deepseek-r1:1.5b`, confirmed pulled locally), and an
`init_db` call so sections 7-8 have tables to write to regardless of whether `run.sh`/the FastAPI
lifespan hook has ever run in this environment. Retitled the notebook's intro to describe both
Phase 1 (sections 1-5) and Phase 2 (sections 6-8) content.

Three new sections: **6. Registry** — lists `TESTS`, printing each entry's dataset
source/config/split, scoring type, and scorer name. **7. Run the full pipeline: MMLU** —
`create_run` + `await run_test(...)` + queries the resulting `runs`/`responses`/`scores` rows
directly (same query pattern used in `test_run_loop.py`), printing each response alongside its
score and rationale. **8. Run the full pipeline: TruthfulQA** — same shape, with `JUDGE_MODEL`
passed through, plus a markdown note pointing at the doc's measurement-honesty section (judge
scores are model opinion, not ground truth) since this is the first place in the notebook a judge
score is actually shown to the user.

Edited via the `NotebookEdit` tool (plain edits aren't permitted on `.ipynb` files, same as
Step 14). Verified the exact new cell logic — not a paraphrase — as an equivalent script against
real Ollama before leaving the committed notebook cells with empty outputs: registry listing
printed both test IDs and their `scoring_type`s correctly, both `run_test` calls (MMLU,
TruthfulQA with the judge) completed with `status == "complete"` at `sample_size=2`. Deleted the
temporary verification database (`/tmp/notebook_phase2_check.db`) afterward. `uv run pytest`:
55/55 still passing (no backend code touched by this step). `uv run ruff check backend`: clean.

## Step 32 — fix notebook validation issue (missing cell `id` fields)

User reported "notebook is not executing successfully." Investigated by actually executing the
notebook end-to-end via `nbclient` (added temporarily via `uv run --with nbclient`, not to the
project's real dependencies) rather than just an equivalent script — closer to what a real
Jupyter kernel does, including top-level `await` handling. Result: **it executed successfully**,
proving the cell logic itself is sound. This does surface one real, if separate, defect worth
fixing regardless: `nbclient`/`nbformat` emitted `MissingIDFieldWarning` — all 14 cells from the
notebook's original creation (Step 10) were missing the `id` field nbformat 4.5+ requires
("this will become a hard error in future nbformat versions"). Fixed by assigning each a unique
id. Also normalized: cells touched by the `NotebookEdit` tool during this session's Phase 2
extension (Step 31) had their `source` written as a single string rather than nbformat's more
common list-of-lines form — both are technically valid per spec, but inconsistent within one
file; unified to list-of-lines for consistency while fixing the ids.

Re-executed via `nbclient` after the fix: succeeds cleanly with no warnings. `uv run pytest`
still 55/55 (no backend code touched).

**Not yet resolved**: the user's original failure. Since the notebook demonstrably executes
correctly both as an equivalent script (verified at Steps 14 and 31) and via a real notebook
kernel (`nbclient`, this step), the cause is most likely environment-specific to their machine —
wrong/no kernel selected in VS Code, or a model referenced in the notebook (`JUDGE_MODEL =
"deepseek-r1:1.5b"`, hardcoded in the Setup cell) not being pulled there. Asked the user for the
exact error message and which cell fails.

(Resolved conversationally, no code change needed: the error was `AttributeError: 'Settings'
object has no attribute 'DB_PATH'` — a stale kernel that had imported `app.core.config` before
`DB_PATH` was added in Phase 2. Fix was restarting the notebook's kernel.)

## Step 33 — dataset inspection: `items_to_dataframe()` (in-memory, not persisted)

User asked for sampled dataset items to be inspectable without an ad-hoc script. First design
(CSV via stdlib `csv.DictWriter`, persisted to `data/dataset_samples/`) was explicitly rejected —
user wanted a pandas DataFrame instead, in-memory during a run/session, not written to disk.

Added `pandas` to `[project] dependencies` in root `pyproject.toml` (not `dev` — this is real
backend code, not test-only). Confirmed low actual cost before adding: `pandas` was already
resolved and installed as a transitive dependency of `datasets` since the very first `uv sync`
during the uv migration (`pandas==3.0.5` in that log's output) — `uv lock && uv sync` after this
change downloaded nothing new, just rebuilt the local `llm-litmus` package.

`backend/app/registry/dataset_loader.py`: new `items_to_dataframe(test, items) -> pd.DataFrame`,
kept fully separate from `load_test_items()` rather than changing that function's return type or
inlining the DataFrame construction into it — `load_test_items` is depended on as-is by
`generate_pass`/`run_test` (Step 28) and covered by already-passing tests; no reason to risk that
for a convenience feature. Callers compose the two:
`items_to_dataframe(test, load_test_items(test, n))`. Columns: the `Test`'s identifying
attributes (`test_id`, `dataset_source`, `dataset_config`, `dataset_split`, `prompt_template`)
followed by each item's raw fields.

`backend/tests/test_dataset_loader.py`: two new tests — column/value correctness against a
hand-built MMLU-shaped item list (pure function, no mocking needed), and the empty-items edge
case. Also verified against the real, already-cached MMLU dataset directly (not mocked):
`items_to_dataframe(TESTS["mmlu_elementary_math"], load_test_items(..., 5))` → shape `(5, 9)`,
columns include `subject` (a real MMLU field this session's hand-built test fixtures never
happened to include, confirming the function handles the actual schema, not just the test's
simplified one). `uv run pytest`: 57/57 passing (55 previous + 2 new). `uv run ruff check
backend`: clean (also caught and fixed one line-length-only issue ruff's default rule set didn't
flag — `E501` isn't in ruff's default selected rules despite `line-length = 100` being
configured, so `line-length` only governs `ruff format`/rules that reference it, not automatic
line-length linting; fixed manually for consistency with the rest of the codebase anyway).

## Step 34 — notebook demo for `items_to_dataframe`

New "## 7. Inspect sampled items as a DataFrame" section in `provider_walkthrough.ipynb`, inserted
between the Registry listing and the MMLU pipeline run (renumbering those two from 7/8 to 8/9,
and updating the notebook's own intro paragraph to match). Loads 5 MMLU items and displays
`items_to_dataframe(...)` directly as a cell's return value — DataFrames render richly in
Jupyter by default, which is the actual "analyze better" benefit over the plain `print()`
statements used elsewhere in this notebook.

Executed the full notebook end-to-end via `nbclient` (the established technique) and saved real
outputs — confirmed the new DataFrame cell renders with real columns
(`test_id`/`dataset_source`/`dataset_config`/`dataset_split`/`prompt_template` plus MMLU's own
`question`/`subject`/`choices`/`answer`) and real sampled rows, not placeholder data. `uv run
pytest`: still 57/57 (no backend code touched by this step).

## Step 35 — rename `update_run_quantization` → `record_run_quantization`

User flagged the name as misleading: `update_run_quantization` implies the app is setting or
controlling quantization, when it only backfills a column with whatever Ollama already reports
via `get_model_info`'s `POST /api/show` — models are pre-quantized at `ollama pull` time, this
function has no influence over that.

Renamed across all 4 real usage sites (confirmed via `grep`, no test referenced the old name
directly): `backend/app/database/db.py` (definition, now with a clarifying docstring spelling out
exactly this — "never selects, requests, or applies a quantization level"),
`backend/app/database/__init__.py` (import + `__all__`), `backend/app/evaluations/run_loop.py`
(import + call site). Pure rename, no behavior change — `uv run pytest` still 57/57 (same tests,
covering the same behavior via `test_run_loop.py`'s existing `target_quantization`/
`judge_quantization` assertions, which don't reference the function by name). `uv run ruff check
backend`: clean. Confirmed via a second `grep` pass that no reference to the old name survives
anywhere in `backend/`.

## Step 36 — quantization docs clarity + backlog entry

Two small, closely-related doc edits, combined into one commit (the plan listed them as separate
steps, but splitting a same-topic two-line doc change across two commits added no real value).

`CLAUDE.md`'s "Provider abstraction" bullet didn't mention `get_model_info` at all — it predates
Phase 2, which is when that method was added. Extended it to name `get_model_info` alongside
`generate_response`/`test_speed`, and added the same clarification as Step 35's docstring: it
reports quantization already baked into a pulled model tag, the app never selects or applies it.
`docs/PROJECT_DOCUMENTATION.md` §2.3's "Model interface" table row got the same clarification
appended to its "Function" column.

Also added item 5 to §"Later phases (beyond Phase 2, in order)": optional expected/preferred
quantization validation (warn or block on mismatch) — explicitly noting Ollama has no
request-time quantization control, so "selecting" quantization in that future feature would mean
picking a different pre-quantized model tag, not runtime conversion. Docs-only; `uv run pytest`
still 57/57.

## Step 37 — diagnosed a real 29-minute run failure; `gather_with_concurrency` helper

A notebook cell failed after 29m28s. Diagnosed via Ollama's own `server.log` (`~/.ollama/logs/`)
and the real `data/llm_litmus.db`, not guesswork: `run_id=12`'s `created_at`→`updated_at` gap
matched the screenshot's duration exactly. Findings, each a real bug:

1. `runs.sample_size` said `5`, but 78 responses were stored (`item_index` 0-77) — the
   notebook's `create_run(..., sample_size=5)` and `run_test(..., sample_size=<much larger>)`
   are two independent literals in the same cell that fell out of sync.
2. Item #79 hit `LocalOllamaProvider`'s fixed 120s timeout — Ollama's log showed it still
   actively decoding past 5,391 tokens (deepseek-r1:1.5b is a reasoning model, prone to long
   chain-of-thought even for simple questions) and it had already triggered a context-window
   shift. Not a down server, just a slow generation exceeding a too-short fixed ceiling.
3. That timeout raised `httpx.ReadTimeout` → caught by the `except httpx.TransportError` clause
   (a `TimeoutException` is a subclass of `TransportError` in httpx) → misreported as
   `OllamaUnreachableError` ("Is it running? Try `ollama serve`"), actively misleading.
4. `run_test` wraps generate+score in one try/except — the failure at item 79 discarded the
   *scoring* of the 78 good responses (0 rows in `scores` for this run) even though, contrary to
   what I initially told the user, the responses themselves were NOT lost — the outer except
   handler's own `conn.commit()` (meant to persist `status="failed"`) incidentally flushed the
   whole pending transaction, including the 78 `insert_response` calls from earlier in the same
   transaction. Correcting my own earlier claim in this log: partial generation is NOT silently
   discarded by the current code, it's just never scored or surfaced as a usable partial result.
5. User separately asked why CPU/GPU sat idle throughout — root cause: `generate_pass` and
   `JudgeScorer.score()` both use `await` inside a sequential loop/list-comprehension, which is
   async *syntax* with zero actual concurrency (one HTTP request in flight at a time, ever).

Fixing all of this as a connected set of changes (see `~/.claude/plans/floofy-cooking-lerdorf.md`
for the full design) — this step is just the first piece: `backend/app/core/concurrency.py`,
`gather_with_concurrency(limit, coros) -> list` — runs coroutines with at most `limit`
concurrent, preserving input order in the results regardless of completion order, returning
exceptions in-place (via `asyncio.gather(..., return_exceptions=True)`) rather than raising, so
one failure doesn't cancel or hide the others.

`backend/tests/test_concurrency.py`: three tests — result ordering holds even when a
slower-first/faster-last mix of delays would expose a bug; peak concurrent execution never
exceeds the configured limit (tracked via a shared counter across 10 tasks bounded to 3); an
exception from one coroutine is returned in-place, not raised, and doesn't affect the others'
results. `uv run pytest`: 60/60 passing (57 previous + 3 new). `uv run ruff check backend`:
clean.

## Step 38 — timeout + concurrency config settings

Added `ollama_request_timeout_seconds` (default `300`) and `ollama_max_concurrent_requests`
(default `3`) to `config.json`, and `OLLAMA_REQUEST_TIMEOUT_SECONDS`/
`OLLAMA_MAX_CONCURRENT_REQUESTS` to `Settings` (`backend/app/core/config.py`) — same
`field(default_factory=...)` + env-override pattern as every other field, established in Step 20
specifically so `Settings()` is re-testable per-instance.

`300` for the timeout is deliberately generous: the diagnosed run's failing item was still
decoding past 5,391 tokens (and had already triggered a context-window shift) when the old 120s
ceiling cut it off — `deepseek-r1:1.5b`'s reasoning traces can run long. `3` for concurrency is
a conservative starting default now that Step 37's helper exists to actually use it (Steps
39-42 wire it in) — user confirmed Ollama's own server-side scheduling auto-adjusts, so this
app's job is just to offer concurrent requests, not to tune server-side parallelism itself.

`backend/tests/test_config.py`: extended (not new) — both new fields added to the existing
defaults-from-config.json test and the env-override test, following the exact established
pattern rather than writing a separate test function per field. `uv run pytest`: still 60/60 (no
new test count — same three test functions, more assertions). `uv run ruff check backend`: clean.

## Step 39 — fix the timeout misclassification; use the configurable timeout

`backend/app/core/providers.py`: new `OllamaTimeoutError(OllamaProviderError)` — states the
model and the timeout value, and explicitly says it may just be a slow/long generation rather
than Ollama being down (the opposite framing of `OllamaUnreachableError`'s "Is it running?").

The actual bug: `except httpx.TransportError` was catching *everything*, including timeouts —
`httpx.TimeoutException` (which covers `ReadTimeout`, `ConnectTimeout`, etc.) is a subclass of
`TransportError` in httpx, so the broader clause silently absorbed the more specific case. Fixed
in both `generate_response` and `get_model_info` by adding `except httpx.TimeoutException`
*before* the existing `except httpx.TransportError` — order matters here since Python checks
except clauses top to bottom; a genuine `ConnectError` still falls through to the unchanged,
correctly-named `OllamaUnreachableError`.

`LocalOllamaProvider.__init__`'s `timeout` default changed from the hardcoded `120.0` to
`settings.OLLAMA_REQUEST_TIMEOUT_SECONDS` (Step 38's new `300`). Verified directly:
`LocalOllamaProvider()._timeout == 300.0`.

`backend/tests/test_providers.py`: two new timeout tests (`generate_response` and
`get_model_info`, mocking `httpx.ReadTimeout`) asserting `OllamaTimeoutError` specifically — and
one explicit regression test mocking `httpx.ConnectError` to confirm that case still raises
`OllamaUnreachableError`, not swallowed by the new timeout handling. `uv run pytest`: 63/63
passing (60 previous + 3 new). `uv run ruff check backend`: clean.

## Step 40 — optional `sample_size` in `load_test_items`

`load_test_items(test, sample_size: int | None = None)` — `None` now means "use the full test":
`n = len(dataset) if sample_size is None else min(sample_size, len(dataset))`. First piece of
the user's explicit request ("if sample size is not provided, it takes the original test size").

`backend/tests/test_dataset_loader.py`: one new test confirming `sample_size=None` returns every
mocked row. Also verified against the real (already-cached) MMLU dataset directly:
`load_test_items(TESTS["mmlu_elementary_math"], sample_size=None)` → 378 items, matching the
count Step 17 confirmed live against the real Hugging Face dataset months ago in this log — a
nice independent cross-check that both findings agree. `uv run pytest`: 64/64 passing (63
previous + 1 new). `uv run ruff check backend`: clean.

## Step 41 — nullable `sample_size` + `record_run_sample_size` + `partial` status

`backend/app/database/schema.py`: `sample_size INTEGER NOT NULL` → `sample_size INTEGER`
(nullable — NULL means "not resolved yet"). Added the `status` column's documented value list as
an inline comment (`pending|generating|scoring|complete|partial|failed`), including the new
`partial` status Step 42 will introduce.

`create_run`'s `sample_size` parameter becomes optional (`int | None = None`), matching the
schema. New `record_run_sample_size(conn, run_id, sample_size)` in `db.py`, structurally
identical to `record_run_quantization` — backfills the actual resolved count once
`load_test_items` runs, same "start unknown, fill in once known" pattern already used for
quantization. Exported from `database/__init__.py`.

`backend/tests/test_database.py`: two new tests — `create_run` without `sample_size` stores
`NULL`; `record_run_sample_size` backfills it correctly (and confirms it's `None` beforehand).
All existing tests (which always pass an explicit `sample_size`) needed no changes — confirms
this is additive, not a behavior change for existing callers. `uv run pytest`: 66/66 passing (64
previous + 2 new). `uv run ruff check backend`: clean.

Note: the real local `data/llm_litmus.db` still has the old `NOT NULL` constraint baked in
(`CREATE TABLE IF NOT EXISTS` doesn't retroactively alter existing tables) — deleting it and
letting `init_db` regenerate it fresh is part of Step 45's real verification.

## Step 42 — concurrent `JudgeScorer.score()`

`score()` previously did `[await self._score_one(item) for item in items]` — sequential, same
zero-concurrency problem as `generate_pass`. Now uses Step 37's `gather_with_concurrency`. No
new failure-handling needed here specifically: `_score_one` already never raises (parse failures
become `JUDGE_PARSE_FAILED` results, not exceptions), so this purely adds concurrency with no
new exception surface to handle. New `concurrency: int = 3` constructor param (Step 43 wires
`settings.OLLAMA_MAX_CONCURRENT_REQUESTS` through via `scorer_factory`).

`backend/tests/test_judge_scorer.py`: one new test with 5 items and `concurrency=2` — the
existing `_fake_provider` helper (canned responses consumed in call order via `side_effect`
list) no longer works once multiple items run concurrently, since call order across *different*
items isn't deterministic anymore. Replaced with a `fake_generate` that inspects the actual
prompt content (each item's `response_text` is distinct) to return the right canned answer
regardless of call order — then asserts `results` come back in the same order as the input
`items` list, not completion order. `uv run pytest`: 67/67 passing (66 previous + 1 new). `uv
run ruff check backend`: clean.

## Step 43 — thread concurrency through `scorer_factory`

`build_scorer`'s judge branch now passes `concurrency=settings.OLLAMA_MAX_CONCURRENT_REQUESTS`
into `JudgeScorer(...)`, so the config value from Step 38 actually reaches the scorer that uses
it (Step 42 added the parameter but nothing was wiring it from config yet).

`backend/tests/test_scorer_factory.py`: extended the existing
`test_build_scorer_returns_judge_for_truthfulqa` with one assertion —
`scorer._concurrency == settings.OLLAMA_MAX_CONCURRENT_REQUESTS` — rather than a new test
function, since it's checking one more property of the same already-tested construction. `uv
run pytest`: still 67/67 (same function count, one more assertion). `uv run ruff check backend`:
clean.

## Step 44 — `run_test`: optional `sample_size`, concurrent `generate_pass`, partial scoring

The core rewrite tying together every prior step in this session's fix.

`generate_pass` no longer loops sequentially — precomputes all prompts, builds all
`generate_response` coroutines, runs them through Step 37's `gather_with_concurrency`
(concurrency from `settings.OLLAMA_MAX_CONCURRENT_REQUESTS`), inserts a response for each
success, and **returns** the list of exceptions from failures instead of raising — so a failure
partway through no longer aborts the items that succeeded alongside it (which, under
concurrency, aren't necessarily a clean prefix anymore — a failure can land anywhere in the
batch).

`run_test`: `sample_size` is now optional (`None` → full test, passed straight through to
`load_test_items`); the actual resolved count is backfilled via `record_run_sample_size`
immediately after loading, before generation even starts, so it's correct even if generation
later fails entirely. After `generate_pass` returns, checks whether *any* response exists
(`get_responses_for_run`):
- **Zero responses** (every item failed) → re-raises the first error, falling into the existing
  outer handler, `status="failed"` — unchanged from before.
- **Any responses exist** → proceeds to `score_pass` on them regardless of the error count, then
  sets `status="complete"` if there were no errors, or the new `status="partial"` (with a
  summary like `"2/4 items failed to generate; first error: ..."` in `runs.error`) if some did.
  This is the direct fix for the diagnosed incident: 78 good responses would have been scored
  and the run marked `partial`, instead of 0 scores and `status="failed"`.

`backend/tests/test_run_loop.py`: the `FakeProvider` had to change from
`iter(target_responses)`/call-order-based dispatch to prompt-content matching (a dict of
substring → canned response, matching the technique already used for `JudgeScorer`'s concurrent
test) — call order across items is no longer meaningful once `generate_pass` runs them
concurrently. Five tests: the two existing end-to-end cases (MMLU pass/fail mix, TruthfulQA with
judge) adapted to the new `FakeProvider`; `sample_size=None` processes a full 6-item mocked
dataset and backfills `sample_size=6`; a **partial failure** case (4 items, 2 fail — deliberately
non-adjacent indices 1 and 3, not a prefix, to prove concurrent/scattered failures are handled)
asserts `status="partial"`, the error summary, and that exactly the 2 successful items (0 and 2)
got both a response and a score; a **total failure** case (both items fail) asserts `status`
stays `"failed"` with zero responses/scores, unchanged behavior. `uv run pytest`: 69/69 passing
(67 previous + 2 net — 3 old tests replaced by 5 new ones). `uv run ruff check backend`: clean
(one line-length-only fix, same `E501`-not-in-default-ruleset situation as Step 33).

Real (unmocked) verification against live Ollama: `run_test` for `mmlu_elementary_math`,
`deepseek-r1:1.5b`, `sample_size=6`, no mocks at all — `status="complete"`, `sample_size=6`
correctly backfilled, 6 responses stored, finished in 75.2s. Faster than the ~90-180s a strictly
sequential run would be expected to take at this model's observed per-item latency (10-30s each,
per Step 37's diagnosis) — consistent with real concurrent execution happening, not just
requests queuing one-at-a-time at Ollama, though a precise apples-to-apples sequential-vs-
concurrent timing comparison on identical items wasn't run.

## Step 45 — optional `sample_size` in the router

`RunTestRequest.sample_size: int` → `int | None = None` in `backend/app/router/runs.py` — the
rest of the endpoint already passed `request.sample_size` straight through to `create_run`/
`run_test`, both already accepting `None` as of Steps 41/44, so no other code changes needed.

`backend/tests/test_runs_router.py`: one new test posting a request that omits `sample_size`
entirely, confirming it's genuinely optional at the API layer (not just defaulted at the Python
call-site level) — asserts `200` and a real `run_id` back. `uv run pytest`: 70/70 passing (69
previous + 1 new). `uv run ruff check backend`: clean.

## Step 46 — notebook: one shared `SAMPLE_SIZE` instead of two literals

The direct fix for what actually caused the incident: the notebook's Setup cell gains
`SAMPLE_SIZE = 5  # None = use the full test`, with a comment explicitly naming the bug it
prevents. Both pipeline sections (MMLU, TruthfulQA) now reference `SAMPLE_SIZE` in both their
`create_run(...)` and `run_test(...)` calls instead of two independent `5`s that could drift
apart — the literal scenario that produced `sample_size=5` recorded against a run that actually
processed 78 items. Also added `sample_size` to both sections' printed output, so the recorded
value is visible without a separate DB query.

Executed the full notebook end-to-end via `nbclient` against real Ollama (`deepseek-r1:1.5b`
target, `llama3.2:latest` judge) — took several minutes given the target model's slow, verbose
reasoning output, ran as a background task while waiting. Both sections completed
`status="complete"` with `sample_size: 5` correctly shown (no drift now — impossible by
construction, not just backfilled-and-correct as Step 44 alone would already have guaranteed),
quantization captured for both models. `uv run pytest`: still 70/70 (no backend code touched).
`uv run ruff check backend`: clean.
