from app.registry import TESTS

# Real sample row captured live from cais/mmlu, elementary_mathematics, test split
# (docs/implementation-log.md Step 17) -- not hand-invented, so this test locks in behavior
# against the actual dataset shape.
MMLU_SAMPLE_ROW = {
    "question": "What is the value of p in 24 = 2p?",
    "choices": ["p = 4", "p = 8", "p = 12", "p = 24"],
    "answer": 2,
}

EXPECTED_MMLU_PROMPT = (
    "Answer the following multiple-choice question. "
    "Respond with only the letter of the correct answer.\n\n"
    "What is the value of p in 24 = 2p?\n\n"
    "A. p = 4\n"
    "B. p = 8\n"
    "C. p = 12\n"
    "D. p = 24\n\n"
    "Answer:"
)


def test_mmlu_elementary_math_registered():
    test = TESTS["mmlu_elementary_math"]
    assert test.id == "mmlu_elementary_math"
    assert test.dataset_source == "cais/mmlu"
    assert test.dataset_config == "elementary_mathematics"
    assert test.dataset_split == "test"
    assert test.scoring_type == "deterministic"
    assert test.scorer_name == "multiple_choice_match"


def test_mmlu_item_to_prompt_renders_exact_text():
    test = TESTS["mmlu_elementary_math"]
    assert test.item_to_prompt(MMLU_SAMPLE_ROW) == EXPECTED_MMLU_PROMPT


def test_mmlu_item_to_gold_maps_int_answer_to_letter():
    test = TESTS["mmlu_elementary_math"]
    # answer=2 -> "ABCD"[2] -> "C", per the real ClassLabel schema confirmed in Step 17.
    assert test.item_to_gold(MMLU_SAMPLE_ROW) == {"answer_letter": "C"}
