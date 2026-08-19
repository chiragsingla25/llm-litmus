import asyncio

import pytest
from app.scorers import BaseScorer, ScoreItem, ScoreResult


def test_base_scorer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseScorer()


def test_score_item_shape():
    item = ScoreItem(response_id=1, response_text="hello", gold_reference={"answer": "A"})
    assert item.response_id == 1
    assert item.response_text == "hello"
    assert item.gold_reference == {"answer": "A"}


def test_score_result_shape_with_defaults():
    result = ScoreResult(response_id=1, score=1.0, passed=True, rationale="matched")
    assert result.raw_judge_output is None


def test_score_result_shape_full():
    result = ScoreResult(
        response_id=2,
        score=0.0,
        passed=False,
        rationale="did not match",
        raw_judge_output='{"score": 0}',
    )
    assert result.raw_judge_output == '{"score": 0}'


def test_concrete_subclass_can_be_instantiated():
    class DummyScorer(BaseScorer):
        async def score(self, items: list[ScoreItem]) -> list[ScoreResult]:
            return [
                ScoreResult(response_id=i.response_id, score=1.0, passed=True, rationale="")
                for i in items
            ]

    scorer = DummyScorer()
    assert isinstance(scorer, BaseScorer)

    items = [ScoreItem(response_id=1, response_text="hi", gold_reference={})]
    results = asyncio.run(scorer.score(items))
    assert results == [ScoreResult(response_id=1, score=1.0, passed=True, rationale="")]
