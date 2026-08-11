from unittest.mock import MagicMock, patch

from app.registry import TESTS
from app.registry.dataset_loader import items_to_dataframe, load_test_items


def _fake_dataset(rows: list[dict]) -> MagicMock:
    fake = MagicMock()
    fake.__getitem__.side_effect = lambda i: rows[i]
    fake.__len__.return_value = len(rows)
    return fake


def test_load_test_items_routes_lm_eval_source():
    rows = [
        {"question": "q0", "lmeval_prompt": "p0", "lmeval_target": "#### 1"},
        {"question": "q1", "lmeval_prompt": "p1", "lmeval_target": "#### 2"},
        {"question": "q2", "lmeval_prompt": "p2", "lmeval_target": "#### 3"},
    ]

    with patch(
        "app.registry.dataset_loader.load_lmeval_task_items", return_value=rows[:2]
    ) as mock_load:
        items = load_test_items(TESTS["lmeval_gsm8k"], sample_size=2)

    mock_load.assert_called_once_with("gsm8k", 2)
    assert items == rows[:2]


def test_load_test_items_lm_eval_mmlu_task_name():
    with patch(
        "app.registry.dataset_loader.load_lmeval_task_items", return_value=[]
    ) as mock_load:
        load_test_items(TESTS["lmeval_mmlu_elementary_mathematics"], sample_size=3)

    mock_load.assert_called_once_with("mmlu_elementary_mathematics", 3)


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


def test_load_test_items_hf_takes_deterministic_first_n_slice():
    rows = [{"question": f"q{i}"} for i in range(10)]

    with patch(
        "app.registry.dataset_loader.load_dataset", return_value=_fake_dataset(rows)
    ):
        items = load_test_items(TESTS["truthfulqa_generation"], sample_size=3)

    assert items == rows[:3]


def test_items_to_dataframe_includes_test_attributes_and_item_fields():
    test = TESTS["lmeval_mmlu_elementary_mathematics"]
    items = [
        {
            "question": "What is the value of p in 24 = 2p?",
            "choices": ["4", "8", "12", "24"],
            "answer": 2,
            "lmeval_prompt": "prompt0",
        },
        {
            "question": "What is 1 + 1?",
            "choices": ["1", "2", "3", "4"],
            "answer": 1,
            "lmeval_prompt": "prompt1",
        },
    ]

    df = items_to_dataframe(test, items)

    assert len(df) == 2
    assert (df["test_id"] == "lmeval_mmlu_elementary_mathematics").all()
    assert (df["dataset_source"] == "lm_eval").all()
    assert (df["dataset_config"] == "mmlu_elementary_mathematics").all()
    assert df.iloc[0]["question"] == "What is the value of p in 24 = 2p?"
    assert df.iloc[0]["answer"] == 2


def test_items_to_dataframe_empty_items():
    df = items_to_dataframe(TESTS["lmeval_gsm8k"], [])
    assert len(df) == 0
