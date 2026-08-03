"""Tests for the minimal runs router -- monkeypatches run_test so nothing here ever touches
Ollama or Hugging Face, and redirects DB_PATH to a tmp_path db so nothing touches the real
project data directory either.
"""

from unittest.mock import AsyncMock, patch

import app.main as main_module
import app.router.runs as runs_module
import pytest
from app.core.config import Settings
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    test_settings = Settings()
    monkeypatch.setattr(main_module, "settings", test_settings)
    monkeypatch.setattr(runs_module, "settings", test_settings)

    with (
        patch.object(runs_module, "run_test", AsyncMock()),
        TestClient(main_module.app) as test_client,
    ):
        yield test_client


def test_trigger_run_unknown_test_id_returns_404(client):
    response = client.post(
        "/tests/run",
        json={"test_id": "not_a_real_test", "target_model": "fake", "sample_size": 2},
    )
    assert response.status_code == 404


def test_trigger_run_success_returns_run_id(client):
    response = client.post(
        "/tests/run",
        json={
            "test_id": "mmlu_elementary_math",
            "target_model": "fake-target",
            "sample_size": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "run_id" in body
    assert isinstance(body["run_id"], int)


def test_trigger_run_without_sample_size_is_accepted(client):
    # Omitting sample_size means "use the full test" -- must not be a required field.
    response = client.post(
        "/tests/run",
        json={"test_id": "mmlu_elementary_math", "target_model": "fake-target"},
    )
    assert response.status_code == 200
    assert isinstance(response.json()["run_id"], int)


def test_get_run_returns_stored_row(client):
    create_response = client.post(
        "/tests/run",
        json={
            "test_id": "mmlu_elementary_math",
            "target_model": "fake-target",
            "sample_size": 2,
        },
    )
    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run_id
    assert body["test_id"] == "mmlu_elementary_math"
    assert body["target_model"] == "fake-target"


def test_get_run_unknown_id_returns_404(client):
    response = client.get("/runs/999999")
    assert response.status_code == 404
