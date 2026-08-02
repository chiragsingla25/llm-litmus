from unittest.mock import MagicMock, patch

from app.registry import TESTS
from app.registry.dataset_loader import load_test_items


def _fake_dataset(rows: list[dict]) -> MagicMock:
    fake = MagicMock()
    fake.__getitem__.side_effect = lambda i: rows[i]
    fake.__len__.return_value = len(rows)
    return fake


def test_load_test_items_takes_deterministic_first_n_slice():
    rows = [{"question": f"q{i}"} for i in range(10)]

    with patch(
        "app.registry.dataset_loader.load_dataset", return_value=_fake_dataset(rows)
    ) as mock_load:
        items = load_test_items(TESTS["mmlu_elementary_math"], sample_size=3)

    mock_load.assert_called_once_with("cais/mmlu", "elementary_mathematics", split="test")
    assert items == rows[:3]


def test_load_test_items_caps_at_dataset_length():
    rows = [{"question": "only one"}]

    with patch("app.registry.dataset_loader.load_dataset", return_value=_fake_dataset(rows)):
        items = load_test_items(TESTS["mmlu_elementary_math"], sample_size=5)

    assert items == rows


def test_load_test_items_uses_truthfulqa_registry_fields():
    rows = [{"question": "q0"}, {"question": "q1"}]

    with patch(
        "app.registry.dataset_loader.load_dataset", return_value=_fake_dataset(rows)
    ) as mock_load:
        items = load_test_items(TESTS["truthfulqa_generation"], sample_size=2)

    mock_load.assert_called_once_with(
        "truthfulqa/truthful_qa", "generation", split="validation"
    )
    assert items == rows
