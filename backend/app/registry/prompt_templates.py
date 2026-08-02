"""Versioned prompt templates for registered tests.

Template names are versioned tags (e.g. "_v1") recorded with every run per CLAUDE.md's
reproducibility requirement -- changing the wording of an existing template means adding a
new version, not editing this one in place.
"""

MMLU_MULTIPLE_CHOICE_V1 = "mmlu_multiple_choice_v1"


def render_mmlu_multiple_choice_v1(question: str, choices: list[str]) -> str:
    lettered = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
    return (
        "Answer the following multiple-choice question. "
        "Respond with only the letter of the correct answer.\n\n"
        f"{question}\n\n{lettered}\n\nAnswer:"
    )
