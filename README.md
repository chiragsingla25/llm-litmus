# Local LLM Evaluation Suite
 
**Project Documentation — Overview & Implementation Plan**
 
A self-hosted tool that runs a local LLM against a small, fixed set of benchmark tests and reports scores. Built around an extensible **Test Registry** so new tests are cheap to add. Two scoring types from day one — deterministic (plain code) and judge-LLM (a second local model) — with only 2–3 tests shipped in v1, on purpose. No API keys, no accounts, no cloud inference anywhere.
 
---
 
## 1. Overview
 
### 1.1 What it is
 
Clone the repo, run one script, point it at a local LLM (via Ollama), and evaluate it against predefined tests pulled from public benchmark datasets. Each test is scored one of two ways:
 
- **Deterministic** — the test has one clear correct answer, checked with plain code.
- **Judge-LLM** — the test has no single right answer; a second local model scores the response against a grounded rubric.
Phase 1 ships 2–3 tests covering both scoring types, built on a Test Registry designed so adding tests later is a new declaration, not new plumbing. All model inference happens on the user's machine. **Ad-hoc evaluation (user types their own prompt) is deliberately a later phase.**
 
### 1.2 Relationship to existing tools (honest framing)
 
The industry-standard tool for this job is **EleutherAI's lm-evaluation-harness** — it powers the Hugging Face Open LLM Leaderboard and runs hundreds of benchmarks against local models. This project is not trying to replace it. It deliberately borrows its good ideas and packages them differently:
 
**Borrowed from lm-eval-harness:** standardized prompt templates per benchmark, separating generation from scoring, recording model/config metadata with every run.
 
**What this project does differently:** a plug-and-play GUI instead of CLI + config files; hardware-aware time budgeting (profile the machine, fit the test run to a user-chosen time budget); judge-LLM scoring using a second local model; and a deliberately tiny, readable codebase — the point is demonstrating architecture and product thinking, not benchmark coverage.
 
Other adjacent tools and their gaps: DeepEval/RAGAs (SDKs, no GUI, judge is typically a cloud API), Promptfoo (CLI matrix testing, static config), Langfuse/Phoenix (observability for deployed apps, not local model testing).
 
### 1.3 Core value pillars
 
- **Zero-key, local-first** — no API keys, no accounts, no sign-ups anywhere in the flow. All inference stays on the machine. (Benchmark datasets are fetched once from Hugging Face — public, ungated, anonymous — then cached locally.)
- **Deliberately small v1** — 2–3 tests. The registry architecture is the deliverable; test count is not.
- **Extensible by construction** — a new test reusing an existing scoring type is one new registry entry.
- **Honest numbers** — sample sizes shown prominently, scores never framed as comparable to published leaderboards, judge scores clearly labeled as model opinion rather than ground truth.
- **Hardware-aware execution** — pre-flight profiling scales test size to the user's machine and time budget.
### 1.4 What this project demonstrates (portfolio signal)
 
- Registry-based extensibility: "what dataset, what scoring type" is cleanly separated from "how the app runs and displays results."
- Two evaluation paradigms implemented properly: deterministic scoring and grounded LLM-as-judge scoring with structured output.
- Async backend (FastAPI + background tasks) handling long-running model calls without blocking the UI.
- Measurement literacy: reproducible decoding settings, recorded model metadata, sample-size caveats — knowing *why* eval numbers can mislead is itself the signal.
- Product discipline: a small finished v1 with a documented roadmap, not a large unfinished one.
---
 
## 2. Architecture
 
### 2.1 System diagram
 
```
+--------------------------------------------------------------------------------+
|                            USER LOCAL WORKSTATION                              |
|                                                                                  |
|  +------------------+    HTTP (polling)    +--------------------------------+  |
|  |   FRONTEND UI    | <------------------> |        FASTAPI BACKEND         |  |
|  | (Next.js/React)  |                      |  +--------------------------+  |  |
|  +------------------+                      |  |  Background Task Runner  |  |  |
|                                            |  +--------------------------+  |  |
|                                            +---------------|----------------+  |
|                                                            |                   |
|                                                            v                   |
|                                              +--------------------------+      |
|                                              |      Test Registry       |      |
|                                              | (dataset + scoring type  |      |
|                                              |  + scorer config)        |      |
|                                              +-------------|------------+      |
|                                                            |                   |
|                              Phase A: generate             |                   |
|                                                            v                   |
|                                              +---------------------------+     |
|                                              |   Provider Interface      |     |
|                                              +-------------|-------------+     |
|                                                            v (localhost:11434) |
|                                              +---------------------------+     |
|                                              |   OLLAMA INFERENCE API    |     |
|                                              | [Target Model] [Judge]    |     |
|                                              +---------------------------+     |
|                                                                                  |
|   Phase B: score stored responses                                               |
|   +-------------------------+     +------------------------------------+       |
|   |  Deterministic scorers  |     |  Judge scorer (batched — judge     |       |
|   |  (plain code, instant)  |     |  model loaded once, not per item)  |       |
|   +-------------------------+     +------------------------------------+       |
|                                                                                  |
|   [ SQLite: runs, responses, scores ]   [ HF dataset cache (fetched once) ]     |
+--------------------------------------------------------------------------------+
```
 
### 2.2 The core design decision: generation and scoring are separate passes
 
Every test run has two phases:
 
- **Phase A — Generate:** pull the test's sample, run every item through the target model, store all raw responses in SQLite.
- **Phase B — Score:** run the stored responses through the test's scorer. Deterministic scorers are instant. Judge scoring runs as its own batch, after all generation is done.
This one decision solves three problems at once:
 
1. **Model swapping.** Consumer GPUs usually can't hold the target and judge models in memory simultaneously; Ollama would reload models on every interleaved call. Batching means each model loads exactly once per run.
2. **Re-scorability.** Raw responses are stored, so a new or improved scorer can re-score old runs without re-running generation — which also makes "add a new scoring type later" genuinely cheap.
3. **Clean progress reporting.** The UI shows two simple progress bars (generating n/N, scoring n/N) instead of one ambiguous one.
### 2.3 Technology stack
 
| Layer | Stack | Function |
|---|---|---|
| Frontend UI | Next.js (React) + Recharts | Test selection, time-budget slider, results dashboard |
| Backend | FastAPI + `asyncio` | API layer, background task orchestration |
| Test Registry | Plain Python declarations (dataclasses or YAML) | The extension point: dataset source, scoring type, scorer config per test |
| Model interface | `BaseLLMProvider` abstraction; `LocalOllamaProvider` is the only Phase 1 implementation | Same interface a cloud provider would implement in a later phase |
| Deterministic scoring | Plain Python | One small function per scorer type (e.g. multiple-choice match) |
| Judge scoring | Structured-output prompting against a user-selected local model | Grounded rubric score + short rationale, JSON-validated with retry-on-parse-failure |
| Dataset sourcing | Hugging Face `datasets` with default local caching (small configs/splits — no streaming complexity needed) | Fetched once anonymously, cached; no token, datasets are public and ungated |
| Storage | SQLite via stdlib `sqlite3` (no ORM) | Runs, raw responses, scores |
| Packaging | Single `run.sh` (Python venv + backend, `npm install` + dev server for frontend) | Clone → run → app opens. macOS/Linux; Windows via WSL (one README line) |
 
### 2.4 The Test Registry
 
A test is a declaration:
 
```
Test = {
  id: "mmlu_elementary_math",
  dataset: { source: "cais/mmlu", config: "elementary_mathematics", split: "test" },
  prompt_template: "mmlu_multiple_choice",     # standardized per benchmark, borrowed practice from lm-eval-harness
  sample_size: "allocator-decided",
  scoring_type: "deterministic",               # or "judge"
  scorer: "multiple_choice_match",             # registered function, or rubric template for judge tests
}
```
 
The run loop, allocator, storage, and dashboard are written against "a list of registered tests" — none of them know about any specific dataset. Adding a test that reuses an existing scoring type is one new entry. Adding a genuinely new scoring type means writing one new scorer function — expected, and still contained.
 
**How the two scoring types are segregated:** they are not two pipelines — they're one pipeline with a single branch point. Both scorer types implement the same small interface (`BaseScorer.score(items) -> list[ScoreResult]`), and the registry entry's `scoring_type` decides which class is instantiated for Phase B. The run loop itself never branches on test type; it just calls `scorer.score()`. Only three places in the codebase know the difference:
 
1. **The scorer classes themselves** — `DeterministicScorer` is a pure function (response + gold answer → pass/fail, instant, no model call); `JudgeScorer` builds a grounded rubric prompt, makes one batched judge-model call per response, and parses a JSON `{score, rationale}` with retry-on-parse-failure.
2. **The allocator** — deterministic items cost 1 model call, judge items cost 2.
3. **The dashboard label** — "accuracy %" vs. "judge score," never blended into one number.
Everything else — registry, dataset fetch, generation pass, storage, run orchestration, progress reporting — is completely type-blind. This is what makes new scoring types cheap: a future `CodeExecutionScorer` (HumanEval) or `F1Scorer` (SQuAD) is one new class implementing the same interface, plus registry entries that reference it.
 
### 2.5 Phase 1 test set
 
| Test | Dataset | Scoring | Notes |
|---|---|---|---|
| MMLU (one subject subset) | `cais/mmlu` (verify ID at build time) | Deterministic | Multiple-choice match — simplest possible proof of the deterministic path |
| TruthfulQA (generation split) | `truthful_qa` | Judge-LLM | The dataset ships `correct_answers` / `incorrect_answers` per question — the judge prompt includes these as reference material, so the judge checks against provided ground truth rather than its own knowledge. Much more reliable than a free-floating "rate this" prompt, and mirrors how the original benchmark's judge worked |
| GSM8K *(stretch only)* | `gsm8k` | Deterministic | Regex-extract final number. Only if time allows; also serves as the proof that adding a registry test is cheap |
 
### 2.6 Measurement honesty (built into the product, not just the README)
 
- **Reproducible decoding:** temperature 0 / greedy for all benchmark runs. Model tag, quantization level, and prompt template version recorded with every run — two runs of the same setup produce the same numbers.
- **Sample-size caveats:** with e.g. 20 questions, the margin of error is large (±10–20pp). The UI displays sample size next to every score and the README states plainly that results are directional, for comparing *your local models against each other* — not comparable to published leaderboard numbers (which use full test sets, specific few-shot formats, and unquantized weights).
- **Prompt-format awareness:** published MMLU numbers use 5-shot prompting; small local models need careful answer-extraction handling. The prompt template per test is explicit and versioned rather than implicit.
- **Judge scores are labeled as model opinion.** Known judge biases (longer/more confident answers, self-preference) are stated in the README.
### 2.7 Hardware-aware time budgeting
 
1. Warm-up prompt profiles tokens/sec for the target model (and judge, if a judge test is selected).
2. User picks a time budget via slider (e.g. 1–15 min).
3. Allocator computes per-test sample sizes: deterministic tests cost 1 generation per item; judge tests cost 1 generation + 1 judge call per item (accounted as two, batched separately).
4. The UI shows the plan (n questions per test, estimated duration) before the user commits.
### 2.8 Setup & run
 
`git clone` → `./run.sh` → dashboard opens. The script: creates a venv and installs backend deps, installs frontend deps, starts both, opens the browser. External requirements: Ollama running with at least one model pulled, plus Node.js. First run fetches and caches the benchmark datasets (the only network access in the entire flow — anonymous, no token). No Docker, no keys, no accounts.
 
---
 
## 3. Implementation Plan
 
**Estimated timeline: 3–5 weeks at 10–15 hrs/week.**
 
### Phase 1 — Foundations (Week 1)
- Repo, project board, branch protection; `CLAUDE.md` for shared Claude Code/Cursor context.
- `BaseLLMProvider` + `LocalOllamaProvider`.
- FastAPI skeleton; `run.sh` working end to end before features exist.
### Phase 2 — Registry, pipeline & scorers (Week 2)
- Test Registry structure + the two Phase 1 test definitions.
- Dataset fetch/cache; standardized prompt templates.
- Two-pass run loop (generate → store → score) via `BackgroundTasks`.
- Deterministic scorer (MMLU) and grounded judge scorer (TruthfulQA, JSON-validated output with retry).
### Phase 3 — Profiling & allocation (Week 3)
- Tokens/sec warm-up profiler; time-budget → sample-size allocator (1x vs 2x call cost per test type).
- `/profile` and `/tests/run` endpoints; run metadata recording (model tag, quantization, template version, decoding params).
### Phase 4 — Dashboard & storage (Week 4)
- Next.js UI: test selector, judge-model selector, budget slider, two-phase progress, results with sample-size display.
- SQLite schema: runs, responses, scores. Historical comparison view.
### Phase 5 — Polish & release (Week 4–5)
- README: setup, architecture diagram, demo GIF, honest-limitations section (judge bias, sample size, quantization), lm-eval-harness acknowledgment, WSL note for Windows.
- Harden `run.sh` error messages (Ollama down, model missing, wrong Python/Node).
- CI: lint + unit tests for allocator math and both scorers. Demo GIF for README.
*(Stretch: GSM8K as a third registry entry — doubles as the proof that adding a test is cheap.)*
 
---
 
## 4. Repository structure
 
```
llm-eval-local/
├── .github/workflows/ci.yml
├── backend/
│   ├── app/
│   │   ├── core/            # providers.py, hardware.py, config
│   │   ├── database/        # sqlite3 helpers (no ORM)
│   │   ├── registry/        # test definitions + prompt templates
│   │   ├── scorers/         # deterministic/, judge/
│   │   ├── evaluations/     # allocator.py, two-pass run orchestration
│   │   ├── router/          # FastAPI endpoints
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   ├── app/                 # Next.js App Router pages
│   ├── components/          # dashboard, test selector, run controls
│   ├── package.json
│   └── next.config.js
├── CLAUDE.md
├── README.md
└── run.sh
```
 
---
 
## 5. Success criteria
 
- `git clone` → `./run.sh` → both test types run successfully with no configuration beyond Ollama + Node.js + a pulled model.
- Runs finish within the selected time budget without freezing the UI.
- Re-running the same model/settings reproduces the same deterministic scores exactly.
- Adding a new test reusing an existing scoring type takes one registry entry — verify this yourself (GSM8K) before calling v1 done.
- No API key, account, or cloud inference anywhere; the only network access is the one-time anonymous dataset fetch.
- README communicates the value, the extensibility, and the honest limitations within a minute of landing on the repo.
---
 
## 6. Next Phase (Phase 2) — concrete scope
 
The immediate next phase after the MVP ships, defined now so v1 architecture decisions are made with it in mind. Phase 2 has two workstreams, in this order:
 
### 6.1 Workstream 1 — Broaden the test suite (pure registry additions)
 
Each of these validates the registry design — none should require touching the run loop, allocator, or dashboard:
 
| New test | Dataset | What it adds |
|---|---|---|
| GSM8K (if not done as v1 stretch) | `gsm8k` | New deterministic scorer: regex numeric extraction |
| SQuAD | `squad` | New deterministic scorer: F1 / exact-match on answer spans |
| HumanEval | `openai_humaneval` | New deterministic scorer: sandboxed code execution (subprocess + timeout, no network) — the most involved scorer, saved for last |
| 2–3 more MMLU subjects | `cais/mmlu` configs | Zero new code — pure registry entries, the cheapest possible additions |
| 1–2 more judge-scored tests | TBD | Reuses `JudgeScorer` with different rubric templates |
 
**Definition of done for this workstream:** at least one contributor-style test addition documented in the README ("how to add a test in ~20 lines"), proving the extension story to anyone reading the repo.
 
### 6.2 Workstream 2 — Ad-hoc evaluation (the deferred v1 feature)
 
User types or pastes any prompt; the target model answers; the judge scores it against a fixed rubric. If the user optionally supplies a reference answer or expected format, a deterministic check runs alongside and is reported separately.
 
Why it's cheap by then: it reuses `JudgeScorer`, the provider interface, storage, and the results UI — the only new pieces are one endpoint (`/evaluate/adhoc`), a prompt-input page, and a history view for one-off evaluations. Estimated at roughly a week, precisely because the v1 architecture already did the hard parts.
 
### Later phases (beyond Phase 2, in order)
 
1. **Cloud provider support** — `CloudOpenAIProvider` / `CloudAnthropicProvider` implementing the existing interface; user-supplied keys entered in the UI (the first time keys appear anywhere in the project).
2. **Fine-tuned dedicated judge model** (stretch) — after the prompted judge's limitations are understood in practice.
3. **Agentic red-teaming** (stretch) — attacker/critic loop probing the target model; separate module.
4. **Live production monitoring** — explicitly considered and deferred; this is a local test runner, not an observability platform.
---
 
## 7. Known Simplifications & Trade-offs
 
- **2–3 tests only, on purpose** — the registry is the deliverable; breadth is roadmap item 1.
- **Sampled results are directional**, for comparing local models against each other — not leaderboard-comparable (sample size, prompt format, quantization all differ from published setups). Stated in the UI, not just the README.
- **The judge is a prompted existing model with known biases** — mitigated by grounding it in the dataset's reference answers, but still model opinion, and labeled as such.
- **Judge tests cost ~2x per item** — handled by batched two-pass execution and reflected in the time-budget math.
- **First run needs internet** for the one-time dataset fetch; cached thereafter. All inference is always local.
- **Two runtimes** (Python + Node) plus Ollama; `run.sh` hides it. Windows requires WSL.
- **Verify HF dataset IDs at build time** — some (e.g. MMLU → `cais/mmlu`) have moved over the years.
---
 
## 8. Open Questions (yours to decide)
 
- **Judge rubric scale for TruthfulQA** — e.g. binary truthful/untruthful vs. 1–5 scale, given the judge sees the reference answer lists. Binary is simpler and easier to aggregate; worth a quick look at how existing judge implementations phrase it.
- **Default judge model** — a sensible UI pre-selection rule (e.g. suggest a model ≥ the target's size) so users don't judge a 8B model with a 1B model.
- **Which MMLU subject(s)** for the deterministic test — pick deliberately (one subject keeps the prompt template simple).
