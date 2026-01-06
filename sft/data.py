"""
Dataset loading and preprocessing for SFT training.

This module handles loading datasets from HuggingFace and converting them
to the standard SFT format (prompt, completion pairs).
"""

from datasets import Dataset, load_dataset


def load_hf_dataset(dataset_id: str, eval_split: float = 0.0) -> tuple[Dataset, Dataset | None]:
    """
    Load dataset from HuggingFace and convert to SFT format.

    Converts dataset to 'prompt' and 'completion' columns for automatic prompt masking.
    Uses a generator for memory-efficient processing.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., 'kiranpg/ocaml-sft-problems')
        eval_split: Fraction of data to use for evaluation (0.0 to 1.0). Default 0.0 (no eval set).

    Returns:
        Tuple of (train_dataset, eval_dataset). eval_dataset is None if eval_split=0.0
    """
    dataset = load_dataset(dataset_id, split="train")

    assert "prompt" in dataset.column_names, f"Dataset must have 'prompt' column, got: {dataset.column_names}"
    assert "solution" in dataset.column_names, f"Dataset must have 'solution' column, got: {dataset.column_names}"

    def to_sft_format(examples: list[dict]) -> Dataset:
        return Dataset.from_generator(
            lambda: ({"prompt": ex["prompt"], "completion": ex["solution"]} for ex in examples)
        )

    if eval_split > 0.0:
        # Shuffle with fixed seed for reproducibility, then split
        dataset = dataset.shuffle(seed=42)
        split_idx = int(len(dataset) * (1 - eval_split))
        train_examples = [dataset[i] for i in range(split_idx)]
        eval_examples = [dataset[i] for i in range(split_idx, len(dataset))]
        return to_sft_format(train_examples), to_sft_format(eval_examples)

    return to_sft_format(list(dataset)), None
