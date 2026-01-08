"""
Dataset loading and preprocessing for SFT training.

This module handles loading datasets from HuggingFace and converting them
to the standard SFT format with proper chat template formatting.

CRITICAL: For instruct models (like Qwen2.5-Coder-Instruct), training data MUST be
formatted using the model's chat template. Without this, the model loses its
instruction-following capability (catastrophic forgetting).
"""

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

# System prompt for OCaml code generation (matches eval/eval.py)
SYSTEM_PROMPT = "Respond only with runnable OCaml code (no prose)."

# User prompt template - instructs model to output in ```ocaml blocks (matching training data)
USER_PROMPT_TEMPLATE = """You are an expert OCaml programmer. Complete the following OCaml function.
Respond with ONLY the function body wrapped in an ```ocaml``` code block.

{problem_text}"""


def format_with_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    problem_text: str,
    solution: str,
) -> dict[str, str]:
    """
    Format a single example using the model's chat template.

    This ensures SFT training uses the SAME format used during inference,
    preventing catastrophic forgetting of instruction-following capability.

    Args:
        tokenizer: The model's tokenizer with chat template
        problem_text: The OCaml function signature/docstring
        solution: The solution code (should be in ```ocaml blocks)

    Returns:
        Dict with 'text' key containing the full formatted conversation
    """
    # Build messages in chat format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem_text=problem_text.strip())},
    ]

    # Apply chat template to get the prompt portion (with generation prompt)
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Full training text = prompt + completion + EOS token
    # Solution should already be in ```ocaml blocks from dataset
    full_text = formatted_prompt + solution.strip() + tokenizer.eos_token

    return {"text": full_text}


def load_hf_dataset(
    dataset_id: str,
    tokenizer: PreTrainedTokenizerBase,
    eval_split: float = 0.0,
) -> tuple[Dataset, Dataset | None]:
    """
    Load dataset from HuggingFace and convert to SFT format with chat template.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., 'kiranpg/ocaml-sft-problems')
        tokenizer: Tokenizer for applying chat template
        eval_split: Fraction of data to use for evaluation (0.0 to 1.0). Default 0.0.

    Returns:
        Tuple of (train_dataset, eval_dataset). eval_dataset is None if eval_split=0.0
    """
    dataset = load_dataset(dataset_id, split="train")

    assert "prompt" in dataset.column_names, f"Dataset must have 'prompt' column, got: {dataset.column_names}"
    assert "solution" in dataset.column_names, f"Dataset must have 'solution' column, got: {dataset.column_names}"

    def to_sft_format(examples: list[dict]) -> Dataset:
        return Dataset.from_generator(
            lambda: (
                format_with_chat_template(tokenizer, ex["prompt"], ex["solution"])
                for ex in examples
            )
        )

    if eval_split > 0.0:
        dataset = dataset.shuffle(seed=42)
        split_idx = int(len(dataset) * (1 - eval_split))
        train_examples = [dataset[i] for i in range(split_idx)]
        eval_examples = [dataset[i] for i in range(split_idx, len(dataset))]
        return to_sft_format(train_examples), to_sft_format(eval_examples)

    return to_sft_format(list(dataset)), None
