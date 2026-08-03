"""JudgeScorer -- generic, JSON-validated, retry-on-parse-failure judge scoring.

Builds a rubric prompt per item, calls a user-selected local judge model, parses a
{"score", "rationale"} JSON response. Never raises on parse failure -- returns a ScoreResult
with passed=None and an explanatory rationale instead, since one bad judge response shouldn't
take down an entire scoring pass.
"""

import json
import re

from app.core.concurrency import gather_with_concurrency
from app.core.providers import BaseLLMProvider
from app.scorers import BaseScorer, ScoreItem, ScoreResult

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. Respond with ONLY a JSON "
    'object of the form {"score": <0 or 1>, "rationale": "<short reason>"}, no other text.'
)


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a judge model's free-form response."""
    candidates = []
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1))
    candidates.append(text)
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        candidates.append(obj_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "score" in parsed:
            return parsed
    return None


class JudgeScorer(BaseScorer):
    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        judge_model: str,
        rubric_template: str,
        max_retries: int = 1,
        concurrency: int = 3,
    ) -> None:
        self._provider = provider
        self._judge_model = judge_model
        self._rubric_template = rubric_template
        self._max_retries = max_retries
        self._concurrency = concurrency

    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]:
        # _score_one never raises (parse failures become JUDGE_PARSE_FAILED results), so no
        # Exception-checking is needed on the results here -- concurrency is the only change.
        return await gather_with_concurrency(
            self._concurrency, [self._score_one(item) for item in items]
        )

    async def _score_one(self, item: ScoreItem) -> ScoreResult:
        prompt = self._rubric_template.format(
            response_text=item.response_text, **item.gold_reference
        )

        raw_output = ""
        for attempt in range(self._max_retries + 1):
            attempt_prompt = prompt if attempt == 0 else prompt + _RETRY_SUFFIX
            result = await self._provider.generate_response(
                self._judge_model, attempt_prompt, temperature=0.0
            )
            raw_output = result.text
            parsed = _extract_json(raw_output)
            if parsed is not None:
                score = float(parsed["score"])
                return ScoreResult(
                    response_id=item.response_id,
                    score=score,
                    passed=score >= 0.5,
                    rationale=str(parsed.get("rationale", "")),
                    raw_judge_output=raw_output,
                )

        return ScoreResult(
            response_id=item.response_id,
            score=None,
            passed=None,
            rationale="JUDGE_PARSE_FAILED",
            raw_judge_output=raw_output,
        )
