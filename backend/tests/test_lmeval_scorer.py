import pytest
from app.scorers import ScoreItem
from app.scorers.integrations.lmeval_scorer import (
    LmEvalGsm8kExactMatchScorer,
    LmEvalMmluMatchScorer,
    extract_gsm8k_number,
)

GOLD_18 = {"answer_number": "18"}
GOLD_C = {"answer_letter": "C"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reasoning...\n#### 18", "18"),
        ("#### 1,234", "1234"),
        ("no hash but ends with 42", "42"),
        ("no numbers here", None),
    ],
)
def test_extract_gsm8k_number(text, expected):
    assert extract_gsm8k_number(text) == expected


GSM8K_CASES = [
    ("#### 18", True),
    ("The answer is #### 18", True),
    ("#### 19", False),
    ("I don't know", False),
]


@pytest.mark.parametrize("response_text,expected_passed", GSM8K_CASES)
def test_gsm8k_exact_match_scorer(response_text, expected_passed):
    scorer = LmEvalGsm8kExactMatchScorer()
    result = scorer.score_one(
        ScoreItem(response_id=1, response_text=response_text, gold_reference=GOLD_18)
    )
    assert result.passed is expected_passed
    assert result.score == (1.0 if expected_passed else 0.0)


def test_gsm8k_missing_number_rationale():
    scorer = LmEvalGsm8kExactMatchScorer()
    result = scorer.score_one(
        ScoreItem(response_id=2, response_text="nope", gold_reference=GOLD_18)
    )
    assert result.passed is False
    assert "no numeric answer found" in result.rationale


def test_lmeval_mmlu_match_scorer_letter():
    scorer = LmEvalMmluMatchScorer()
    result = scorer.score_one(
        ScoreItem(response_id=1, response_text="Answer: C", gold_reference=GOLD_C)
    )
    assert result.passed is True
    assert result.score == 1.0
