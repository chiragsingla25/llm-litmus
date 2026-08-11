"""Deterministic scorers aligned with lm-eval GSM8K / MMLU generative conventions.

These score *stored* Phase A responses only — they never call lm_eval.simple_evaluate().
"""

from __future__ import annotations

import re

from app.scorers import ScoreItem, ScoreResult
from app.scorers.deterministic import DeterministicScorer, MultipleChoiceMatchScorer

# lm-eval GSM8K marks the final numeric answer after ####.
_GSM8K_HASH_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.\d+)?)")
_GSM8K_NUMBER_RE = re.compile(r"-?[0-9][0-9,]*(?:\.\d+)?")


def extract_gsm8k_number(text: str) -> str | None:
    """Prefer ####-marked answer (lm-eval convention); else last number in the text."""
    hash_matches = _GSM8K_HASH_RE.findall(text)
    if hash_matches:
        return _normalize_number(hash_matches[-1])
    numbers = _GSM8K_NUMBER_RE.findall(text.replace(",", ""))
    if not numbers:
        return None
    return _normalize_number(numbers[-1])


def _normalize_number(raw: str) -> str:
    return raw.replace(",", "").strip()


class LmEvalGsm8kExactMatchScorer(DeterministicScorer):
    """Exact-match on the final numeric answer (lm-eval GSM8K #### convention)."""

    def score_one(self, item: ScoreItem) -> ScoreResult:
        gold = str(item.gold_reference["answer_number"])
        extracted = extract_gsm8k_number(item.response_text)

        if extracted is None:
            return ScoreResult(
                response_id=item.response_id,
                score=0.0,
                passed=False,
                rationale="no numeric answer found in response",
            )

        passed = extracted == gold
        return ScoreResult(
            response_id=item.response_id,
            score=1.0 if passed else 0.0,
            passed=passed,
            rationale=f"extracted '{extracted}', gold '{gold}'",
        )


class LmEvalMmluMatchScorer(MultipleChoiceMatchScorer):
    """Letter match for lm-eval MMLU; reuses the shared A–D extraction heuristics."""
