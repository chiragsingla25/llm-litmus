"""build_scorer() -- the one place, besides the scorer classes themselves and the dashboard
label, allowed to branch on a Test's scoring_type/scorer_name, per
docs/PROJECT_DOCUMENTATION.md section 2.4.
"""

from app.core.providers import BaseLLMProvider
from app.registry import Test
from app.scorers import BaseScorer
from app.scorers.deterministic import MultipleChoiceMatchScorer
from app.scorers.judge import JudgeScorer
from app.scorers.judge.truthfulqa import TRUTHFULQA_RUBRIC_V1

_DETERMINISTIC_SCORERS: dict[str, type[MultipleChoiceMatchScorer]] = {
    "multiple_choice_match": MultipleChoiceMatchScorer,
}

_JUDGE_RUBRICS: dict[str, str] = {
    "truthfulqa_judge": TRUTHFULQA_RUBRIC_V1,
}


def build_scorer(
    test: Test,
    *,
    provider: BaseLLMProvider | None = None,
    judge_model: str | None = None,
) -> BaseScorer:
    if test.scoring_type == "deterministic":
        scorer_cls = _DETERMINISTIC_SCORERS.get(test.scorer_name)
        if scorer_cls is None:
            raise ValueError(f"Unknown deterministic scorer: {test.scorer_name!r}")
        return scorer_cls()

    if test.scoring_type == "judge":
        rubric = _JUDGE_RUBRICS.get(test.scorer_name)
        if rubric is None:
            raise ValueError(f"Unknown judge scorer: {test.scorer_name!r}")
        if provider is None or judge_model is None:
            raise ValueError("Judge scoring requires both provider and judge_model")
        return JudgeScorer(provider=provider, judge_model=judge_model, rubric_template=rubric)

    raise ValueError(f"Unknown scoring_type: {test.scoring_type!r}")
