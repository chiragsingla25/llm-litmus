"""Test Registry: declarative test definitions.

The run loop, allocator, storage, and dashboard iterate this registry and never branch on a
specific test/dataset name -- see CLAUDE.md. scorer_name is a string key resolved later by
scorer_factory.build_scorer(), never a class reference, so this module stays pure data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.registry.prompt_templates import MMLU_MULTIPLE_CHOICE_V1, render_mmlu_multiple_choice_v1


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


def _mmlu_item_to_prompt(row: dict[str, Any]) -> str:
    return render_mmlu_multiple_choice_v1(row["question"], row["choices"])


def _mmlu_item_to_gold(row: dict[str, Any]) -> dict[str, Any]:
    # MMLU's `answer` field is a ClassLabel int (0-3), not a letter -- confirmed against the
    # real dataset schema in docs/implementation-log.md Step 17.
    return {"answer_letter": "ABCD"[row["answer"]]}


TESTS: dict[str, Test] = {
    "mmlu_elementary_math": Test(
        id="mmlu_elementary_math",
        dataset_source="cais/mmlu",
        dataset_config="elementary_mathematics",
        dataset_split="test",
        prompt_template=MMLU_MULTIPLE_CHOICE_V1,
        scoring_type="deterministic",
        scorer_name="multiple_choice_match",
        item_to_prompt=_mmlu_item_to_prompt,
        item_to_gold=_mmlu_item_to_gold,
    ),
}
