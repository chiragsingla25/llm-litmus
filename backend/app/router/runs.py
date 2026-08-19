"""Minimal endpoints to trigger and inspect test runs.

Deliberately minimal per Phase 2's scope boundary: no listing, no cancellation, no allocator
integration. create_run() is called synchronously so the caller gets a run_id back immediately;
quantization capture and everything else happens inside run_test(), scheduled via
BackgroundTasks, so this endpoint never itself waits on Ollama.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.providers import LocalOllamaProvider
from app.database import create_run, get_connection, get_run
from app.evaluations.run_loop import run_test
from app.registry import TESTS

router = APIRouter()


class RunTestRequest(BaseModel):
    test_id: str
    target_model: str
    sample_size: int | None = None
    judge_model: str | None = None


@router.post("/tests/run")
def trigger_run(request: RunTestRequest, background_tasks: BackgroundTasks) -> dict:
    test = TESTS.get(request.test_id)
    if test is None:
        raise HTTPException(status_code=404, detail=f"Unknown test_id: {request.test_id!r}")

    with get_connection(settings.DB_PATH) as conn:
        run_id = create_run(
            conn,
            test_id=request.test_id,
            target_model=request.target_model,
            judge_model=request.judge_model,
            prompt_template_version=test.prompt_template,
            sample_size=request.sample_size,
            status="pending",
        )

    provider = LocalOllamaProvider()
    background_tasks.add_task(
        run_test,
        request.test_id,
        target_model=request.target_model,
        judge_model=request.judge_model,
        sample_size=request.sample_size,
        provider=provider,
        db_path=settings.DB_PATH,
        run_id=run_id,
    )

    return {"run_id": run_id}


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: int) -> dict:
    with get_connection(settings.DB_PATH) as conn:
        run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return dict(run)
