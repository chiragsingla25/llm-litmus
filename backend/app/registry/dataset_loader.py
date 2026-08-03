"""Dataset fetch/cache -- pulls a registered test's items from Hugging Face.

Uses `datasets`' default local caching (no streaming mode), per CLAUDE.md. The only network
access in the whole flow, and only on first fetch of a given dataset/config/split.
"""

import pandas as pd
from datasets import load_dataset

from app.registry import Test


def load_test_items(test: Test, sample_size: int | None = None) -> list[dict]:
    """Fetches a test's items. `sample_size=None` means "use the full test" -- every row in
    the dataset's registered split."""
    dataset = load_dataset(test.dataset_source, test.dataset_config, split=test.dataset_split)
    # Deterministic first-N slice -- no seed dependency, so re-running the same
    # model/settings reproduces the same sample, per CLAUDE.md's reproducibility rule.
    n = len(dataset) if sample_size is None else min(sample_size, len(dataset))
    return [dataset[i] for i in range(n)]


def items_to_dataframe(test: Test, items: list[dict]) -> pd.DataFrame:
    """Builds a DataFrame of sampled items enriched with the Test's identifying attributes,
    for interactive inspection/analysis (e.g. in a notebook) -- in-memory only, not persisted.
    """
    records = [
        {
            "test_id": test.id,
            "dataset_source": test.dataset_source,
            "dataset_config": test.dataset_config,
            "dataset_split": test.dataset_split,
            "prompt_template": test.prompt_template,
            **item,
        }
        for item in items
    ]
    return pd.DataFrame(records)
