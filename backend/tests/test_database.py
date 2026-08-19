import json
import sqlite3

import pytest
from app.database.db import (
    create_run,
    get_connection,
    get_responses_for_run,
    get_run,
    init_db,
    insert_response,
    insert_score,
    record_run_sample_size,
    update_run_status,
)


def _init_db(db_path):
    with get_connection(db_path) as conn:
        init_db(conn)
    return db_path


def test_init_db_creates_expected_tables(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"runs", "responses", "scores"} <= tables


def test_foreign_keys_pragma_is_on(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        (fk_enabled,) = conn.execute("PRAGMA foreign_keys").fetchone()

    assert fk_enabled == 1


def test_duplicate_item_index_for_same_run_is_rejected(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (test_id, target_model, prompt_template_version, sample_size) "
            "VALUES ('t', 'm', 'v1', 1)"
        )
        run_id = conn.execute("SELECT id FROM runs").fetchone()["id"]
        conn.execute(
            "INSERT INTO responses (run_id, item_index, prompt, response_text, gold_reference) "
            "VALUES (?, 0, 'p', 'r', '{}')",
            (run_id,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO responses (run_id, item_index, prompt, response_text, gold_reference) "
                "VALUES (?, 0, 'p2', 'r2', '{}')",
                (run_id,),
            )


def test_run_response_score_round_trip(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        run_id = create_run(
            conn,
            test_id="lmeval_gsm8k",
            target_model="llama3.2:latest",
            target_quantization="Q4_K_M",
            prompt_template_version="lmeval_gsm8k_v1",
            sample_size=2,
        )

        # Insert out of natural order to prove get_responses_for_run sorts by item_index.
        response_id_1 = insert_response(
            conn,
            run_id=run_id,
            item_index=1,
            prompt="second prompt",
            response_text="B",
            gold_reference={"answer_letter": "B"},
        )
        response_id_0 = insert_response(
            conn,
            run_id=run_id,
            item_index=0,
            prompt="first prompt",
            response_text="A",
            gold_reference={"answer_letter": "C"},
        )

        insert_score(
            conn,
            response_id=response_id_0,
            scorer_version="multiple_choice_match_v1",
            score=0.0,
            passed=False,
        )
        insert_score(
            conn,
            response_id=response_id_1,
            scorer_version="multiple_choice_match_v1",
            score=1.0,
            passed=True,
        )

        update_run_status(conn, run_id, "complete")

        responses = get_responses_for_run(conn, run_id)
        run = get_run(conn, run_id)
        scores = conn.execute("SELECT * FROM scores ORDER BY id").fetchall()

    assert [r["item_index"] for r in responses] == [0, 1]
    assert json.loads(responses[0]["gold_reference"]) == {"answer_letter": "C"}
    assert run["status"] == "complete"
    assert run["target_quantization"] == "Q4_K_M"
    assert [s["passed"] for s in scores] == [0, 1]


def test_update_run_status_records_error(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        run_id = create_run(
            conn,
            test_id="t",
            target_model="m",
            prompt_template_version="v1",
            sample_size=1,
        )
        update_run_status(conn, run_id, "failed", error="boom")
        run = get_run(conn, run_id)

    assert run["status"] == "failed"
    assert run["error"] == "boom"


def test_get_run_returns_none_for_missing_id(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        run = get_run(conn, 999)

    assert run is None


def test_create_run_without_sample_size_stores_null(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        run_id = create_run(
            conn, test_id="t", target_model="m", prompt_template_version="v1"
        )
        run = get_run(conn, run_id)

    assert run["sample_size"] is None


def test_record_run_sample_size_backfills_actual_count(tmp_path):
    db_path = _init_db(tmp_path / "test.db")

    with get_connection(db_path) as conn:
        run_id = create_run(
            conn, test_id="t", target_model="m", prompt_template_version="v1"
        )
        assert get_run(conn, run_id)["sample_size"] is None

        record_run_sample_size(conn, run_id, 378)
        run = get_run(conn, run_id)

    assert run["sample_size"] == 378
