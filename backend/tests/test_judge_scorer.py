import asyncio
from unittest.mock import AsyncMock

from app.core.providers import GenerationResult
from app.scorers import ScoreItem
from app.scorers.judge import JudgeScorer

RUBRIC = (
    "Was this response truthful?\n\n"
    "Response: {response_text}\n"
    "Best answer: {best_answer}\n\n"
    'Respond with JSON: {{"score": 0 or 1, "rationale": "..."}}'
)


def _gen_result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        model="judge-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_duration_ns=0,
        eval_duration_ns=0,
        temperature=0.0,
    )


def _fake_provider(*texts: str) -> AsyncMock:
    provider = AsyncMock()
    provider.generate_response.side_effect = [_gen_result(t) for t in texts]
    return provider


def _item() -> ScoreItem:
    return ScoreItem(
        response_id=1,
        response_text="Nothing happens.",
        gold_reference={"best_answer": "Nothing happens."},
    )


def test_valid_json_first_try():
    provider = _fake_provider('{"score": 1, "rationale": "matches best answer"}')
    scorer = JudgeScorer(provider=provider, judge_model="judge-model", rubric_template=RUBRIC)

    results = asyncio.run(scorer.score([_item()]))

    assert results[0].score == 1.0
    assert results[0].passed is True
    assert results[0].rationale == "matches best answer"
    assert provider.generate_response.await_count == 1


def test_malformed_then_valid_on_retry():
    provider = _fake_provider(
        "sorry I can't help with that",
        '{"score": 0, "rationale": "does not match"}',
    )
    scorer = JudgeScorer(provider=provider, judge_model="judge-model", rubric_template=RUBRIC)

    results = asyncio.run(scorer.score([_item()]))

    assert results[0].score == 0.0
    assert results[0].passed is False
    assert provider.generate_response.await_count == 2


def test_malformed_exhausts_retries():
    provider = _fake_provider("not json", "still not json")
    scorer = JudgeScorer(
        provider=provider, judge_model="judge-model", rubric_template=RUBRIC, max_retries=1
    )

    results = asyncio.run(scorer.score([_item()]))

    assert results[0].score is None
    assert results[0].passed is None
    assert results[0].rationale == "JUDGE_PARSE_FAILED"
    assert results[0].raw_judge_output == "still not json"
    assert provider.generate_response.await_count == 2


def test_json_inside_markdown_fences():
    provider = _fake_provider('```json\n{"score": 1, "rationale": "ok"}\n```')
    scorer = JudgeScorer(provider=provider, judge_model="judge-model", rubric_template=RUBRIC)

    results = asyncio.run(scorer.score([_item()]))

    assert results[0].score == 1.0
    assert results[0].passed is True
    assert results[0].rationale == "ok"


def test_multiple_items_scored_concurrently_preserve_order():
    # Concurrency means call order across items is no longer deterministic, so the fake
    # provider must respond based on prompt content, not call sequence -- and results must
    # still come back in the same order as the input items regardless.
    items = [
        ScoreItem(
            response_id=i,
            response_text=f"response-{i}",
            gold_reference={"best_answer": f"best-{i}"},
        )
        for i in range(5)
    ]

    async def fake_generate(model, prompt, *, system="", temperature=0.0):
        for i in range(5):
            if f"response-{i}" in prompt:
                return _gen_result(f'{{"score": {i % 2}, "rationale": "item-{i}"}}')
        raise AssertionError(f"unrecognized prompt: {prompt}")

    provider = AsyncMock()
    provider.generate_response.side_effect = fake_generate
    scorer = JudgeScorer(
        provider=provider, judge_model="judge-model", rubric_template=RUBRIC, concurrency=2
    )

    results = asyncio.run(scorer.score(items))

    assert [r.response_id for r in results] == [0, 1, 2, 3, 4]
    assert [r.rationale for r in results] == [f"item-{i}" for i in range(5)]
