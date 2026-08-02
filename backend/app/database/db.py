"""sqlite3 connection, init, and CRUD helpers. No ORM, per CLAUDE.md."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.database.schema import SCHEMA_SQL


@contextmanager
def get_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def create_run(
    conn: sqlite3.Connection,
    *,
    test_id: str,
    target_model: str,
    prompt_template_version: str,
    sample_size: int,
    target_quantization: str | None = None,
    judge_model: str | None = None,
    judge_quantization: str | None = None,
    temperature: float = 0.0,
    status: str = "pending",
) -> int:
    cursor = conn.execute(
        "INSERT INTO runs (test_id, target_model, target_quantization, judge_model, "
        "judge_quantization, prompt_template_version, temperature, sample_size, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            test_id,
            target_model,
            target_quantization,
            judge_model,
            judge_quantization,
            prompt_template_version,
            temperature,
            sample_size,
            status,
        ),
    )
    return cursor.lastrowid


def update_run_status(
    conn: sqlite3.Connection, run_id: int, status: str, *, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, error = ?, updated_at = datetime('now') WHERE id = ?",
        (status, error, run_id),
    )


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def update_run_quantization(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    target_quantization: str | None = None,
    judge_quantization: str | None = None,
) -> None:
    conn.execute(
        "UPDATE runs SET target_quantization = ?, judge_quantization = ? WHERE id = ?",
        (target_quantization, judge_quantization, run_id),
    )


def insert_response(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    item_index: int,
    prompt: str,
    response_text: str,
    gold_reference: dict,
) -> int:
    cursor = conn.execute(
        "INSERT INTO responses (run_id, item_index, prompt, response_text, gold_reference) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, item_index, prompt, response_text, json.dumps(gold_reference)),
    )
    return cursor.lastrowid


def get_responses_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM responses WHERE run_id = ? ORDER BY item_index", (run_id,)
    ).fetchall()


def insert_score(
    conn: sqlite3.Connection,
    *,
    response_id: int,
    scorer_version: str,
    score: float | None,
    passed: bool | None,
    rationale: str | None = None,
    raw_judge_output: str | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO scores (response_id, scorer_version, score, passed, rationale, "
        "raw_judge_output) VALUES (?, ?, ?, ?, ?, ?)",
        (
            response_id,
            scorer_version,
            score,
            None if passed is None else int(passed),
            rationale,
            raw_judge_output,
        ),
    )
    return cursor.lastrowid
