from unittest.mock import MagicMock, patch

from app.registry import TESTS
from app.registry.dataset_loader import items_to_dataframe, load_test_items


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


def test_load_test_items_none_sample_size_returns_full_dataset():
    rows = [{"question": f"q{i}"} for i in range(12)]

    with patch("app.registry.dataset_loader.load_dataset", return_value=_fake_dataset(rows)):
        items = load_test_items(TESTS["mmlu_elementary_math"], sample_size=None)

    assert items == rows
    assert len(items) == 12


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


def test_items_to_dataframe_includes_test_attributes_and_item_fields():
    test = TESTS["mmlu_elementary_math"]
    items = [
        {
            "question": "What is the value of p in 24 = 2p?",
            "choices": ["4", "8", "12", "24"],
            "answer": 2,
        },
        {"question": "What is 1 + 1?", "choices": ["1", "2", "3", "4"], "answer": 1},
    ]

    df = items_to_dataframe(test, items)

    assert len(df) == 2
    assert list(df.columns) == [
        "test_id",
        "dataset_source",
        "dataset_config",
        "dataset_split",
        "prompt_template",
        "question",
        "choices",
        "answer",
    ]
    assert (df["test_id"] == "mmlu_elementary_math").all()
    assert (df["dataset_source"] == "cais/mmlu").all()
    assert df.iloc[0]["question"] == "What is the value of p in 24 = 2p?"
    assert df.iloc[0]["answer"] == 2
    assert df.iloc[1]["question"] == "What is 1 + 1?"


def test_items_to_dataframe_empty_items():
    df = items_to_dataframe(TESTS["mmlu_elementary_math"], [])
    assert len(df) == 0
