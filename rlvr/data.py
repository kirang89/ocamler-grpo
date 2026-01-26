"""Dataset loading and preprocessing for RLVR/GRPO training.

This module provides:
- PROMPT_TEMPLATE for formatting OCaml problems
- Dataset loading from HuggingFace or local CSV
- Tokenizer configuration for GRPO generation
"""

import csv
import os
import textwrap

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

DEFAULT_TRAINING_DATASET = "kiranpg/ocaml-training-problems"

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an expert OCaml engineer. Complete the function body below by following this EXACT format:

    1. Start with <code>
    2. Write ONLY the function body (the code that comes after the = sign)
    3. End with </code>
    4. Do NOT repeat the function signature - it is already provided
    5. Do NOT explain the code or include ANY text before or after the code tags

    Examples (for instruction only — do NOT copy these into your answer):

    Problem:
    (**Filter positive numbers from a list
     * >>> filter_positive [1; -2; 3; -4]
     * [1; 3]
    *)
    let filter_positive (numbers : int list) : int list =
    Solution:
    <code>
    List.filter (fun x -> x > 0) numbers
    </code>

    Problem:
    (**Count occurrences of a character in a string
     * >>> count_char "hello" 'l'
     * 2
    *)
    let count_char (s : string) (c : char) : int =
    Solution:
    <code>
    String.fold_left (fun acc ch -> if ch = c then acc + 1 else acc) 0 s
    </code>

    Problem:
    (**Calculate the sum of all elements in a list
     * >>> sum_list [1; 2; 3]
     * 6
    *)
    let rec sum_list (lst : int list) : int =
    Solution:
    <code>
    match lst with
    | [] -> 0
    | head :: tail -> head + sum_list tail
    </code>

    Now solve this problem and complete the provided function:

    Problem ({problem_id}):
    {question}
    """
).strip()


def build_training_dataset(dataset_id: str) -> Dataset:
    """Load a Hugging Face dataset or CSV file and format it for GRPO training.

    Args:
        dataset_id: Either a Hugging Face dataset identifier (e.g., 'username/dataset-name')
                   or a local path to a CSV file (for backwards compatibility)

    Returns:
        A Hugging Face Dataset with formatted prompts for each problem
    """
    # Check if it's a local CSV file (backwards compatibility)
    if dataset_id.endswith(".csv") and os.path.exists(dataset_id):
        with open(dataset_id, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        source = f"local CSV {dataset_id}"
    else:
        # Use "force_redownload" to always re-download and show progress
        hf_dataset = load_dataset(
            dataset_id, split="train", download_mode="reuse_dataset_if_exists"
        )
        rows = [dict(row) for row in hf_dataset]
        source = f"HF dataset {dataset_id}"

    # Process all rows into training format
    dataset_rows = []
    for row in rows:
        problem_id = row["id"]
        question = row["prompt"]  # Column is named "prompt"
        tests = row["tests"]
        prompt = PROMPT_TEMPLATE.format(problem_id=problem_id, question=question)
        dataset_rows.append(
            {"prompt": prompt, "problem_id": problem_id, "question": question, "tests": tests}
        )

    if not dataset_rows:
        raise ValueError(f"No rows found in dataset: {dataset_id}")

    print(f"Loaded {len(dataset_rows)} training problems from {source}")
    return Dataset.from_list(dataset_rows)


def create_tokenizer(model_id: str):
    """Load a tokenizer configured for GRPO generation.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        AutoTokenizer configured with left padding for generation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer
