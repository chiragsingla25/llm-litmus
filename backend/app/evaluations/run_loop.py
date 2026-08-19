"""Two-pass run orchestration.

Phase A (generate_pass) runs every item through the target model and stores raw responses.
Phase B (score_pass) runs the stored responses through the test's scorer. The two phases never
interleave -- per CLAUDE.md, this means each model (target, then judge) loads into Ollama
exactly once per run, instead of reloading on every alternating call.
"""

import json
import sqlite3
from pathlib import Path

from app.core.concurrency import gather_with_concurrency
from app.core.config import settings
from app.core.providers import BaseLLMProvider
from app.database import (
    get_connection,
    get_responses_for_run,
    insert_response,
    insert_score,
    record_run_quantization,
    record_run_sample_size,
    update_run_status,
)
from app.evaluations.scorer_factory import build_scorer
from app.registry import TESTS, Test
from app.registry.dataset_loader import load_test_items
from app.scorers import ScoreItem


async def generate_pass(
    conn: sqlite3.Connection,
    *,
    test: Test,
    items: list[dict],
    target_model: str,
    provider: BaseLLMProvider,
    run_id: int,
    concurrency: int = 3,
) -> list[Exception]:
    """Runs every item through the target model concurrently (bounded by `concurrency`).
    Successful items are inserted as responses; failures are returned, not raised, so the
    caller can score whatever succeeded instead of discarding it."""
    prompts = [test.item_to_prompt(row) for row in items]
    coros = [
        provider.generate_response(target_model, prompt, temperature=0.0) for prompt in prompts
    ]
    results = await gather_with_concurrency(concurrency, coros)

    errors: list[Exception] = []
    for index, (row, prompt, result) in enumerate(zip(items, prompts, results)):
        if isinstance(result, Exception):
            errors.append(result)
            continue
        insert_response(
            conn,
            run_id=run_id,
            item_index=index,
            prompt=prompt,
            response_text=result.text,
            gold_reference=test.item_to_gold(row),
        )
    return errors


async def score_pass(
    conn: sqlite3.Connection,
    *,
    test: Test,
    run_id: int,
    provider: BaseLLMProvider | None = None,
    judge_model: str | None = None,
) -> None:
    responses = get_responses_for_run(conn, run_id)
    scorer = build_scorer(test, provider=provider, judge_model=judge_model)

    items = [
        ScoreItem(
            response_id=row["id"],
            response_text=row["response_text"],
            gold_reference=json.loads(row["gold_reference"]),
        )
        for row in responses
    ]
    results = await scorer.score(items)
    for result in results:
        insert_score(
            conn,
            response_id=result.response_id,
            scorer_version=test.scorer_name,
            score=result.score,
            passed=result.passed,
            rationale=result.rationale,
            raw_judge_output=result.raw_judge_output,
        )


async def run_test(
    test_id: str,
    *,
    target_model: str,
    provider: BaseLLMProvider,
    db_path: Path,
    run_id: int,
    sample_size: int | None = None,
    judge_model: str | None = None,
) -> None:
    """Orchestrates one full run. Never raises -- failures are recorded on the run row
    (status="failed"/"partial" + error message) since this is meant to be scheduled as a
    background task, where nothing downstream could catch an exception anyway.

    sample_size=None means "use the full test." If some (but not all) items fail to generate,
    the run still scores whatever succeeded and finishes status="partial" rather than
    discarding it -- only a total failure (nothing generated at all) is status="failed"."""
    test = TESTS[test_id]

    with get_connection(db_path) as conn:
        try:
            target_info = await provider.get_model_info(target_model)
            judge_info = await provider.get_model_info(judge_model) if judge_model else None
            record_run_quantization(
                conn,
                run_id,
                target_quantization=target_info.quantization_level,
                judge_quantization=judge_info.quantization_level if judge_info else None,
            )
            conn.commit()

            update_run_status(conn, run_id, "generating")
            conn.commit()

            items = load_test_items(test, sample_size)
            record_run_sample_size(conn, run_id, len(items))
            conn.commit()

            errors = await generate_pass(
                conn,
                test=test,
                items=items,
                target_model=target_model,
                provider=provider,
                run_id=run_id,
                concurrency=settings.OLLAMA_MAX_CONCURRENT_REQUESTS,
            )
            conn.commit()

            responses_exist = bool(get_responses_for_run(conn, run_id))
            if errors and not responses_exist:
                # Every item failed -- nothing to score, genuinely failed.
                raise errors[0]

            update_run_status(conn, run_id, "scoring")
            conn.commit()

            await score_pass(
                conn,
                test=test,
                run_id=run_id,
                provider=provider,
                judge_model=judge_model,
            )

            if errors:
                final_status = "partial"
                final_error = (
                    f"{len(errors)}/{len(items)} items failed to generate; "
                    f"first error: {errors[0]}"
                )
            else:
                final_status = "complete"
                final_error = None
            update_run_status(conn, run_id, final_status, error=final_error)
            conn.commit()
        except Exception as exc:  # noqa: BLE001 -- top-level task runner, must record any
            # failure (provider, dataset fetch, scorer, DB) as a failed run rather than let
            # it vanish into FastAPI's BackgroundTasks with nothing downstream to catch it.
            update_run_status(conn, run_id, "failed", error=str(exc))
            conn.commit()
