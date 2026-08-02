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
