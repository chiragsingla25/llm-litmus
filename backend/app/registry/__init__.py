"""Test Registry: declarative test definitions.

The run loop, allocator, storage, and dashboard iterate this registry and never branch on a
specific test/dataset name -- see CLAUDE.md. scorer_name is a string key resolved later by
scorer_factory.build_scorer(), never a class reference, so this module stays pure data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.registry.prompt_templates import (
    LMEVAL_GSM8K_V1,
    LMEVAL_MMLU_ELEMENTARY_MATHEMATICS_V1,
    TRUTHFULQA_GENERATION_V1,
    render_truthfulqa_generation_v1,
)


@dataclass(frozen=True)
class Test:
    id: str
    dataset_source: str
    dataset_config: str
    dataset_split: str
    prompt_template: str
    scoring_type: str
    scorer_name: str
    item_to_prompt: Callable[[dict[str, Any]], str]
    item_to_gold: Callable[[dict[str, Any]], dict[str, Any]]


def _lmeval_item_to_prompt(row: dict[str, Any]) -> str:
    return row["lmeval_prompt"]


def _gsm8k_gold_number(target: str) -> str:
    """Normalize lm-eval GSM8K target (full solution with #### N) to the final number."""
    marker = "####"
    if marker in target:
        return target.split(marker)[-1].strip().replace(",", "")
    raise ValueError(f"Could not extract GSM8K gold number from target: {target!r}")


def _lmeval_gsm8k_item_to_gold(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("lmeval_target", row.get("answer", ""))
    return {"answer_number": _gsm8k_gold_number(str(target))}


def _lmeval_mmlu_item_to_gold(row: dict[str, Any]) -> dict[str, Any]:
    # lm-eval MMLU docs use ClassLabel int 0-3; generative target may also be a letter.
    answer = row.get("answer")
    if isinstance(answer, int):
        return {"answer_letter": "ABCD"[answer]}
    target = row.get("lmeval_target")
    if isinstance(target, int):
        return {"answer_letter": "ABCD"[target]}
    if isinstance(target, str) and target.strip() in "ABCD":
        return {"answer_letter": target.strip()}
    raise ValueError(f"Could not map MMLU gold from row answer={answer!r} target={target!r}")


def _truthfulqa_item_to_prompt(row: dict[str, Any]) -> str:
    return render_truthfulqa_generation_v1(row["question"])


def _truthfulqa_item_to_gold(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "best_answer": row["best_answer"],
        "correct_answers": row["correct_answers"],
        "incorrect_answers": row["incorrect_answers"],
    }


TESTS: dict[str, Test] = {
    "lmeval_gsm8k": Test(
        id="lmeval_gsm8k",
        dataset_source="lm_eval",
        dataset_config="gsm8k",
        dataset_split="test",
        prompt_template=LMEVAL_GSM8K_V1,
        scoring_type="deterministic",
        scorer_name="lmeval_gsm8k_exact_match",
        item_to_prompt=_lmeval_item_to_prompt,
        item_to_gold=_lmeval_gsm8k_item_to_gold,
    ),
    "lmeval_mmlu_elementary_mathematics": Test(
        id="lmeval_mmlu_elementary_mathematics",
        dataset_source="lm_eval",
        dataset_config="mmlu_elementary_mathematics",
        dataset_split="test",
        prompt_template=LMEVAL_MMLU_ELEMENTARY_MATHEMATICS_V1,
        scoring_type="deterministic",
        scorer_name="lmeval_mmlu_match",
        item_to_prompt=_lmeval_item_to_prompt,
        item_to_gold=_lmeval_mmlu_item_to_gold,
    ),
    "truthfulqa_generation": Test(
        id="truthfulqa_generation",
        dataset_source="truthfulqa/truthful_qa",
        dataset_config="generation",
        dataset_split="validation",
        prompt_template=TRUTHFULQA_GENERATION_V1,
        scoring_type="judge",
        scorer_name="truthfulqa_judge",
        item_to_prompt=_truthfulqa_item_to_prompt,
        item_to_gold=_truthfulqa_item_to_gold,
    ),
}
