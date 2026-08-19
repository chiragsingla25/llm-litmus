from unittest.mock import MagicMock, patch

import pytest
from app.registry.lmeval_loader import load_lmeval_task_items


def test_load_lmeval_task_items_enriches_docs_and_slices():
    docs = [
        {"question": "q0", "answer": "#### 1"},
        {"question": "q1", "answer": "#### 2"},
        {"question": "q2", "answer": "#### 3"},
    ]
    fake_task = MagicMock()
    fake_task.test_docs.return_value = docs
    fake_task.doc_to_text.side_effect = lambda d: f"PROMPT:{d['question']}"
    fake_task.doc_to_target.side_effect = lambda d: d["answer"]

    fake_manager = MagicMock()
    fake_manager.load.return_value = {"tasks": {"gsm8k": fake_task}}

    with patch("app.registry.lmeval_loader.TaskManager", return_value=fake_manager):
        items = load_lmeval_task_items("gsm8k", sample_size=2)

    assert len(items) == 2
    assert items[0]["question"] == "q0"
    assert items[0]["lmeval_prompt"] == "PROMPT:q0"
    assert items[0]["lmeval_target"] == "#### 1"
    assert items[1]["lmeval_prompt"] == "PROMPT:q1"
    fake_manager.load.assert_called_once_with(["gsm8k"])


def test_load_lmeval_task_items_unknown_task_raises():
    fake_manager = MagicMock()
    fake_manager.load.return_value = {"tasks": {}}

    with (
        patch("app.registry.lmeval_loader.TaskManager", return_value=fake_manager),
        pytest.raises(ValueError, match="Unknown lm-eval task"),
    ):
        load_lmeval_task_items("not_a_task", sample_size=1)
