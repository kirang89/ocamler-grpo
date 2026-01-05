"""
Dataset loading and preprocessing for SFT training.

This module handles loading datasets from HuggingFace and converting them
to the standard SFT format (prompt, completion pairs).
"""

from datasets import Dataset, load_dataset


def _prompt_completion_generator(dataset: Dataset):
    """Yield prompt-completion pairs from the dataset."""
    for example in dataset:
        yield {"prompt": example["prompt"], "completion": example["solution"]}


def load_hf_dataset(dataset_id: str) -> Dataset:
    """
    Load dataset from HuggingFace and convert to SFT format.

    Converts dataset to 'prompt' and 'completion' columns for automatic prompt masking.
    Uses a generator for memory-efficient processing.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., 'kiranpg/ocaml-sft-problems')

    Returns:
        Dataset with 'prompt' and 'completion' columns ready for SFTTrainer
    """
    dataset = load_dataset(dataset_id, split="train")

    assert "prompt" in dataset.column_names, f"Dataset must have 'prompt' column, got: {dataset.column_names}"
    assert "solution" in dataset.column_names, f"Dataset must have 'solution' column, got: {dataset.column_names}"

    return Dataset.from_generator(lambda: _prompt_completion_generator(dataset))
