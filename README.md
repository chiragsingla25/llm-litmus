# llm-litmus

**A litmus test for your local LLMs.** Benchmark any Ollama model on your own machine — no API keys, no accounts, no cloud, no Docker. Clone, run one script, get scores.

<!-- DEMO GIF GOES HERE (Phase 5): terminal `./run.sh` → dashboard opens → test runs → radar chart -->
<!-- ![demo](docs/demo.gif) -->

## What it does

Point llm-litmus at a local model and it runs a curated set of benchmark tests, scored two ways:

- **Deterministic tests** — questions with one correct answer (e.g. MMLU multiple choice), checked in plain code. Exact, reproducible accuracy numbers.
- **Judge-LLM tests** — questions with no single right answer (e.g. TruthfulQA), scored by a second local model of your choice against a rubric grounded in the dataset's reference answers. Score + written rationale per item.

Before running anything, it profiles your hardware (tokens/sec) and fits the test size to a time budget you pick — a 5-minute run on a MacBook Air and a 5-minute run on a 4090 both finish in 5 minutes, just with different sample sizes.

## Quickstart

**Requirements:** [Ollama](https://ollama.com) running with at least one model pulled, Node.js 18+, Python 3.11+. macOS/Linux (Windows via WSL).

```bash
git clone https://github.com/chiragsingla25/llm-litmus.git
cd llm-litmus
./run.sh
```

The dashboard opens in your browser. Pick a target model, optionally a judge model, set your time budget, hit run.

First run downloads small benchmark dataset samples from Hugging Face (public, no token needed) and caches them. After that, everything works offline. **No inference ever leaves your machine.**

## How it works

```
Frontend (Next.js) ──polling──> FastAPI backend
                                     |
                              Test Registry  (dataset + scoring type + scorer, per test)
                                     |
                    Phase A: generate — target model answers every question (Ollama)
                    Phase B: score   — deterministic scorers run instantly;
                                       judge model scores its batch in one pass
                                     |
                              SQLite (runs, raw responses, scores)
```

Generation and scoring are separate passes on purpose: your GPU only ever holds one model at a time (no thrashing model reloads), and raw responses are stored — so future scorers can re-score old runs without re-running the model.

## Adding a test

Tests are declarative registry entries. If your test reuses an existing scoring type, it's ~10 lines and no changes anywhere else:

```python
Test(
    id="mmlu_world_history",
    dataset=DatasetRef(source="cais/mmlu", config="high_school_world_history", split="test"),
    prompt_template="mmlu_multiple_choice",
    scoring_type="deterministic",
    scorer="multiple_choice_match",
)
```

New scoring type needed? Implement one class against `BaseScorer.score(items) -> list[ScoreResult]` and reference it from your registry entry. The run loop, allocator, and dashboard never need to know.

## Honest limitations

- **Scores are directional, not leaderboard-comparable.** Sampled runs (e.g. 20 questions) carry a large margin of error, published benchmarks use different prompt formats and full test sets, and Ollama models are quantized. Use llm-litmus to compare *your* local models against each other under identical conditions — sample sizes are shown next to every score.
- **The judge is a prompted model, not ground truth.** LLM judges have known biases (longer answers, self-preference). Grounding the rubric in the dataset's reference answers mitigates this; it doesn't eliminate it. Judge scores are labeled separately from deterministic accuracy, never blended.
- **Small test suite, by design.** v1 ships a handful of tests on an architecture built for adding more. If you want hundreds of benchmarks and don't need a GUI, use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — this project borrows several of its practices (standardized prompt templates, generation/scoring separation, per-run metadata).

## Reproducibility

All benchmark generation runs at temperature 0. Every run records the model tag, quantization level, prompt-template version, and decoding parameters — the same model and settings produce the same deterministic scores, every time.

## Roadmap

More tests via the registry (GSM8K, SQuAD, HumanEval), ad-hoc prompt evaluation, cloud provider support (user-supplied keys), and beyond — see the full [design doc](docs/PROJECT_PLAN.md) for the phased plan and the reasoning behind the architecture.

## License

MIT
