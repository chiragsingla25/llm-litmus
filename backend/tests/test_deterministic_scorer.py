import pytest
from app.scorers import ScoreItem
from app.scorers.deterministic import MultipleChoiceMatchScorer

GOLD_C = {"answer_letter": "C"}

CASES = [
    ("C", GOLD_C, True, "bare letter, exact match"),
    ("D", GOLD_C, False, "bare letter, wrong answer"),
    ("The answer is C.", GOLD_C, True, "explicit 'answer is' phrasing"),
    ("Answer: C", GOLD_C, True, "explicit 'Answer:' phrasing"),
    ("(C)", GOLD_C, True, "parenthesized letter"),
    ("c", GOLD_C, True, "lowercase letter, case-insensitive"),
    (
        "Options A and B don't apply, so the answer is C.",
        GOLD_C,
        True,
        "reasoning mentions other letters before explicit answer",
    ),
    ("I think it might be A or maybe C", GOLD_C, True, "no explicit phrasing, falls back to last letter"),
    ("There is no correct choice here.", GOLD_C, False, "no letter present at all"),
]


@pytest.mark.parametrize("response_text,gold,expected_passed,description", CASES, ids=[c[3] for c in CASES])
def test_multiple_choice_match_scorer(response_text, gold, expected_passed, description):
    scorer = MultipleChoiceMatchScorer()
    item = ScoreItem(response_id=1, response_text=response_text, gold_reference=gold)

    result = scorer.score_one(item)

    assert result.passed is expected_passed
    assert result.score == (1.0 if expected_passed else 0.0)
    assert result.response_id == 1


def test_no_match_has_explanatory_rationale():
    scorer = MultipleChoiceMatchScorer()
    item = ScoreItem(response_id=2, response_text="no letter here", gold_reference=GOLD_C)

    result = scorer.score_one(item)

    assert result.passed is False
    assert "no answer letter found" in result.rationale
