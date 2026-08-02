"""Dataset fetch/cache -- pulls a registered test's items from Hugging Face.

Uses `datasets`' default local caching (no streaming mode), per CLAUDE.md. The only network
access in the whole flow, and only on first fetch of a given dataset/config/split.
"""

from datasets import load_dataset

from app.registry import Test


def load_test_items(test: Test, sample_size: int) -> list[dict]:
    dataset = load_dataset(test.dataset_source, test.dataset_config, split=test.dataset_split)
    # Deterministic first-N slice -- no seed dependency, so re-running the same
    # model/settings reproduces the same sample, per CLAUDE.md's reproducibility rule.
    n = min(sample_size, len(dataset))
    return [dataset[i] for i in range(n)]
