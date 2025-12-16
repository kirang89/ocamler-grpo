import csv
import ctypes
import os
import sys
import textwrap
from pathlib import Path

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from logger import RewardLogger, log_learning_metrics
from reward import RewardEvaluator, build_reward_functions


def _ensure_cuda_driver():
    """
    Attempt to locate and preload libcuda.so.1 if it's not automatically found.
    This is common on cloud instances where the driver is in a non-standard path
    or LD_LIBRARY_PATH isn't set in the python environment.
    """
    if sys.platform != "linux":
        return
    # Common search paths for libcuda.so.1 on Linux
    search_paths = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    ]
    # Try to load the library
    for path in search_paths:
        if os.path.exists(path):
            try:
                # RTLD_GLOBAL ensures symbols are visible to subsequently loaded libraries (like torch)
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_ensure_cuda_driver()

import torch

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an expert OCaml engineer. Provide a solution to the problem below by following this EXACT format:

    1. Start with ```ocaml
    2. Write ONLY the OCaml code solution
    3. End with ```
    4. Do NOT include more than one code fence
    4. Do NOT include ANY text before or after the code fence
    5. Do NOT include explanations, comments outside code, or conversational text

    Examples(for instruction only — do NOT copy these into your answer):

    Problem: Filter positive numbers from a list
    ```ocaml
    let filter_positive (numbers : int list) : int list =
      List.filter (fun x -> x > 0) numbers
    ```

    Problem: Count occurrences of a character in a string
    ```ocaml
    let count_char (s : string) (c : char) : int =
      String.fold_left (fun acc ch -> if ch = c then acc + 1 else acc) 0 s
    ```

    Problem: Calculate the sum of all elements in a list
    ```ocaml
    let rec sum_list (lst : int list) : int =
      match lst with
      | [] -> 0
      | head :: tail -> head + sum_list tail
    ```

    Now solve this problem:

    Problem ({problem_id}):
    {question}
    """
).strip()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAINING_DATASET = os.environ.get("TRAINING_DATASET", "kiranpg/ocaml-training-problems")
GRPO_OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", "grpo_runs")


class LearningMetricsCallback(TrainerCallback):
    """Callback that logs essential learning metrics using log_learning_metrics."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is not None:
            log_learning_metrics(self.log_path, logs)


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
    """Load a tokenizer configured for GRPO generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def create_grpo_config(temperature=None) -> GRPOConfig:
    """Assemble GRPO training defaults plus any overrides from env vars.
    Note: These settings have been optimized for running on a RTX 6000 48 GB VRAM.
    """
    # Detect CUDA availability
    cuda_available = torch.cuda.is_available()
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    # set to 4 prompts/step if VRAM allows; reduce when using larger models.
    per_device_batch = int(os.environ.get("GRPO_BATCH_SIZE", "4"))
    # Leave at 1 with batch 4; raise to 2-4 only when you must drop batch size.
    grad_steps = int(os.environ.get("GRPO_GRAD_ACCUM_STEPS", "1"))
    # Target 4 completions/prompt for the RTX box—turn this up until you near 44 GB VRAM.
    num_generations = int(os.environ.get("GRPO_NUM_GENERATIONS", "4"))
    # Increase to ~512 tokens on the RTX rig to capture full OCaml problems.
    max_prompt = int(os.environ.get("GRPO_MAX_PROMPT", "512"))
    # Mirror completions at ~512 tokens so solutions + harnesses fit.
    max_completion = int(os.environ.get("GRPO_MAX_COMPLETION", "512"))
    # Stick with 1-2 passes; GRPO overfits small OCaml sets quickly.
    num_epochs = float(os.environ.get("GRPO_NUM_EPOCHS", "1"))
    # 5e-6 trains safely; bump toward 8e-6 only if the run is stable.
    learning_rate = float(os.environ.get("GRPO_LEARNING_RATE", "5e-6"))
    # Use KL Coefficient to penalize large policy shifts from reference model
    beta = float(os.environ.get("GRPO_BETA", "0.0"))

    generation_batch_size = int(
        os.environ.get("GRPO_GENERATION_BATCH_SIZE", str(per_device_batch * num_generations))
    )

    if temperature is None:
        temperature = float(os.environ.get("GRPO_TEMPERATURE", "0.7"))

    # Gradient clipping to prevent training instability
    max_grad_norm = float(os.environ.get("GRPO_MAX_GRAD_NORM", "1.0"))

    # Optional: Entropy-based token filtering (focuses training on high-entropy tokens)
    # Based on "Beyond the 80/20 Rule" paper - using 20% of highest entropy tokens
    # achieves similar performance to all tokens while improving efficiency
    top_entropy_quantile = float(os.environ.get("GRPO_TOP_ENTROPY_QUANTILE", "0.2"))

    return GRPOConfig(
        temperature=temperature,  # for training diversity
        top_p=0.95,
        output_dir=GRPO_OUTPUT_DIR,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_steps,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        max_prompt_length=max_prompt,
        max_completion_length=max_completion,
        remove_unused_columns=False,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,  # Clip gradients to prevent explosions
        log_completions=True,  # Important for detecting reward collapse
        # Keep it 1 or 2 – frequent logging helps spot reward collapse
        logging_steps=int(os.environ.get("GRPO_LOGGING_STEPS", "1")),
        bf16=use_bf16,  # Auto-detect bf16 support based on CUDA availability
        # Disable checkpointing to avoid requires_grad issues on RTX 6000 training.
        gradient_checkpointing=False,
        eval_strategy="no",
        save_steps=100,
        dataloader_num_workers=8,  # Use CPU cores
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        beta=beta,
        top_entropy_quantile=top_entropy_quantile,  # Focus training on high-entropy tokens
    )


def create_lora_config() -> LoraConfig:
    """Build a LoraConfig using optional env overrides."""
    # Rank 16 keeps VRAM in check; double it only if you need more adapter capacity.
    lora_r = int(os.environ.get("LORA_R", "32"))
    # Alpha 32 pairs well with r=16; scale roughly 2x the rank when you change it.
    lora_alpha = int(os.environ.get("LORA_ALPHA", "64"))
    # Small dropout (5%) stabilizes GRPO; set to 0 if you notice underfitting.
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    # Bias "none" avoids extra params; use "lora_only" when the base model expects it.
    bias = os.environ.get("LORA_BIAS", "none")
    # Cover attention (q/k/v/o) plus MLP (gate/up/down) blocks for coder backbones.
    raw_target_modules = os.environ.get(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    target_modules = [module.strip() for module in raw_target_modules.split(",") if module.strip()]

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def resolve_model_id() -> str:
    """Return a Hugging Face model identifier suitable for GRPO training."""
    candidate = os.environ.get("GRPO_MODEL_ID") or os.environ.get("HF_MODEL_ID")
    if candidate:
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("GRPO_MODEL_ID was provided but empty.")
        if ":" in candidate:
            raise ValueError(
                f"GRPO_MODEL_ID must be a Hugging Face repo id (no ':' characters). Got: {candidate}"
            )
        return candidate
    return DEFAULT_MODEL_ID


def main():
    model_id = resolve_model_id()
    dataset = build_training_dataset(TRAINING_DATASET)
    tokenizer = create_tokenizer(model_id)
    evaluator = RewardEvaluator()

    output_path = Path(GRPO_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_logger = RewardLogger(output_path / "reward_logs")
    reward_funcs = build_reward_functions(evaluator, reward_logger)

    config = create_grpo_config()
    lora_config = create_lora_config()

    # Create learning metrics callback
    learning_log_path = output_path / "learning.log"
    learning_callback = LearningMetricsCallback(learning_log_path)

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[learning_callback],
    )

    trainer.train()

    trainer.save_model(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)


if __name__ == "__main__":
    main()
