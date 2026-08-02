import sqlite3

import pytest
from app.database.db import get_connection, init_db


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
