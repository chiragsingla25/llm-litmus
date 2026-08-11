from app.registry import TESTS

# Shape matching lm-eval mmlu_elementary_mathematics docs after lmeval_loader enrichment.
LMEVAL_MMLU_SAMPLE_ROW = {
    "question": "What is the value of p in 24 = 2p?",
    "choices": ["p = 4", "p = 8", "p = 12", "p = 24"],
    "answer": 2,
    "lmeval_prompt": (
        "What is the value of p in 24 = 2p?\n"
        "A. p = 4\n"
        "B. p = 8\n"
        "C. p = 12\n"
        "D. p = 24\n"
        "Answer:"
    ),
    "lmeval_target": 2,
}

LMEVAL_GSM8K_SAMPLE_ROW = {
    "question": "Janet’s ducks lay 16 eggs per day. ...",
    "answer": "She makes $18 every day.\n#### 18",
    "lmeval_prompt": "Question: Janet’s ducks ...\nAnswer:",
    "lmeval_target": "She makes $18 every day.\n#### 18",
}


def test_lmeval_mmlu_registered():
    test = TESTS["lmeval_mmlu_elementary_mathematics"]
    assert test.id == "lmeval_mmlu_elementary_mathematics"
    assert test.dataset_source == "lm_eval"
    assert test.dataset_config == "mmlu_elementary_mathematics"
    assert test.dataset_split == "test"
    assert test.scoring_type == "deterministic"
    assert test.scorer_name == "lmeval_mmlu_match"


def test_lmeval_gsm8k_registered():
    test = TESTS["lmeval_gsm8k"]
    assert test.dataset_source == "lm_eval"
    assert test.dataset_config == "gsm8k"
    assert test.scorer_name == "lmeval_gsm8k_exact_match"


def test_lmeval_mmlu_item_to_prompt_uses_loader_prompt():
    test = TESTS["lmeval_mmlu_elementary_mathematics"]
    assert test.item_to_prompt(LMEVAL_MMLU_SAMPLE_ROW) == LMEVAL_MMLU_SAMPLE_ROW["lmeval_prompt"]


def test_lmeval_mmlu_item_to_gold_maps_int_answer_to_letter():
    test = TESTS["lmeval_mmlu_elementary_mathematics"]
    assert test.item_to_gold(LMEVAL_MMLU_SAMPLE_ROW) == {"answer_letter": "C"}


def test_lmeval_gsm8k_item_to_gold_extracts_hash_number():
    test = TESTS["lmeval_gsm8k"]
    assert test.item_to_gold(LMEVAL_GSM8K_SAMPLE_ROW) == {"answer_number": "18"}
