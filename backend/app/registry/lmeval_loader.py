"""Load benchmark items from lm-evaluation-harness tasks.

Phase A still generates via our Ollama provider; this module only fetches questions/gold
from lm-eval task definitions (never calls lm_eval.simple_evaluate).
"""

from __future__ import annotations

from typing import Any

from lm_eval.tasks import TaskManager


def load_lmeval_task_items(task_name: str, sample_size: int | None = None) -> list[dict[str, Any]]:
    """Returns the first N docs from an lm-eval task's test split.

    Each item includes the original doc fields plus:
    - `lmeval_prompt`: task.doc_to_text(doc) for Phase A
    - `lmeval_target`: task.doc_to_target(doc) for gold normalization
    """
    task_manager = TaskManager()
    loaded = task_manager.load([task_name])
    try:
        task = loaded["tasks"][task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown lm-eval task: {task_name!r}") from exc

    docs = list(task.test_docs())
    n = len(docs) if sample_size is None else min(sample_size, len(docs))
    items: list[dict[str, Any]] = []
    for i in range(n):
        doc = dict(docs[i])
        doc["lmeval_prompt"] = task.doc_to_text(doc)
        doc["lmeval_target"] = task.doc_to_target(doc)
        items.append(doc)
    return items
