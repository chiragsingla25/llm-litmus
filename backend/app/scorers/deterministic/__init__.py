"""DeterministicScorer implementations. Pure code, no model calls."""

import re
from abc import abstractmethod

from app.scorers import BaseScorer, ScoreItem, ScoreResult


class DeterministicScorer(BaseScorer):
    """Wraps a synchronous score_one() in the async score() loop required by BaseScorer."""

    @abstractmethod
    def score_one(self, item: ScoreItem) -> ScoreResult: ...

    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]:
        return [self.score_one(item) for item in items]


# Prefer an explicit "answer is X" / "Answer: X" phrasing when present -- models often reason
# through multiple options before concluding, so a bare first-letter match would misfire on
# text like "Options A and B are wrong, the answer is C".
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+)?answer(?:\s+is)?\s*[:\s]\s*\(?([A-Da-d])\)?\b", re.IGNORECASE
)
_BARE_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def _extract_answer_letter(response_text: str) -> str | None:
    match = _EXPLICIT_ANSWER_RE.search(response_text)
    if match is None:
        # No explicit phrasing found -- fall back to the last standalone A-D letter, on the
        # heuristic that a model's final answer tends to appear at the end of its response.
        matches = _BARE_LETTER_RE.findall(response_text)
        if not matches:
            return None
        match_text = matches[-1]
    else:
        match_text = match.group(1)
    return match_text.upper()


class MultipleChoiceMatchScorer(DeterministicScorer):
    """Extracts an answer letter (A-D) from free-form model text and compares to gold."""

    def score_one(self, item: ScoreItem) -> ScoreResult:
        gold_letter = item.gold_reference["answer_letter"]
        extracted = _extract_answer_letter(item.response_text)

        if extracted is None:
            return ScoreResult(
                response_id=item.response_id,
                score=0.0,
                passed=False,
                rationale="no answer letter found in response",
            )

        passed = extracted == gold_letter
        return ScoreResult(
            response_id=item.response_id,
            score=1.0 if passed else 0.0,
            passed=passed,
            rationale=f"extracted '{extracted}', gold '{gold_letter}'",
        )
