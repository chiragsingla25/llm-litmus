"""Versioned prompt templates for registered tests.

Template names are versioned tags (e.g. "_v1") recorded with every run per CLAUDE.md's
reproducibility requirement -- changing the wording of an existing template means adding a
new version, not editing this one in place.

lm-eval-backed tests record the task's doc_to_text formatting under these version tags;
the actual prompt string is produced by lm-eval at load time (see lmeval_loader).
"""

# lm-eval task formatting version tags (prompt body comes from task.doc_to_text).
LMEVAL_GSM8K_V1 = "lmeval_gsm8k_v1"
LMEVAL_MMLU_ELEMENTARY_MATHEMATICS_V1 = "lmeval_mmlu_elementary_mathematics_v1"

TRUTHFULQA_GENERATION_V1 = "truthfulqa_generation_v1"


def render_truthfulqa_generation_v1(question: str) -> str:
    # TruthfulQA's own methodology asks the target model the question directly, with no
    # multiple-choice scaffolding -- grounding against correct_answers/incorrect_answers
    # happens later, in the judge's rubric prompt (JudgeScorer), not here.
    return question
