"""BaseScorer interface lives here.

Both scorer types (DeterministicScorer, JudgeScorer) implement the same async interface so
the run loop never branches on scoring_type -- it just calls scorer.score(items).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreItem:
    response_id: int
    response_text: str
    gold_reference: dict[str, Any]


@dataclass(frozen=True)
class ScoreResult:
    response_id: int
    score: float | None
    passed: bool | None
    rationale: str | None
    raw_judge_output: str | None = None


class BaseScorer(ABC):
    @abstractmethod
    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]: ...
