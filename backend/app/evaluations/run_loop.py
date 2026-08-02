"""Two-pass run orchestration.

Phase A (generate_pass) runs every item through the target model and stores raw responses.
Phase B (score_pass) runs the stored responses through the test's scorer. The two phases never
interleave -- per CLAUDE.md, this means each model (target, then judge) loads into Ollama
exactly once per run, instead of reloading on every alternating call.
"""

import json
import sqlite3
from pathlib import Path

from app.core.providers import BaseLLMProvider
from app.database import (
    get_connection,
    get_responses_for_run,
    insert_response,
    insert_score,
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
) -> None:
    for index, row in enumerate(items):
        prompt = test.item_to_prompt(row)
        gold_reference = test.item_to_gold(row)
        result = await provider.generate_response(target_model, prompt, temperature=0.0)
        insert_response(
            conn,
            run_id=run_id,
            item_index=index,
            prompt=prompt,
            response_text=result.text,
            gold_reference=gold_reference,
        )


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
    sample_size: int,
    provider: BaseLLMProvider,
    db_path: Path,
    run_id: int,
    judge_model: str | None = None,
) -> None:
    """Orchestrates one full run. Never raises -- failures are recorded on the run row
    (status="failed" + error message) since this is meant to be scheduled as a background
    task, where nothing downstream could catch an exception anyway."""
    test = TESTS[test_id]

    with get_connection(db_path) as conn:
        try:
            update_run_status(conn, run_id, "generating")
            conn.commit()

            items = load_test_items(test, sample_size)
            await generate_pass(
                conn,
                test=test,
                items=items,
                target_model=target_model,
                provider=provider,
                run_id=run_id,
            )
            conn.commit()

            update_run_status(conn, run_id, "scoring")
            conn.commit()

            await score_pass(
                conn,
                test=test,
                run_id=run_id,
                provider=provider,
                judge_model=judge_model,
            )

            update_run_status(conn, run_id, "complete")
            conn.commit()
        except Exception as exc:  # noqa: BLE001 -- top-level task runner, must record any
            # failure (provider, dataset fetch, scorer, DB) as a failed run rather than let
            # it vanish into FastAPI's BackgroundTasks with nothing downstream to catch it.
            update_run_status(conn, run_id, "failed", error=str(exc))
            conn.commit()
