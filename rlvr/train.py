#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for OCaml code generation.

Uses TRL's GRPOTrainer with:
- Verifiers-based reward functions
- LoRA for efficient fine-tuning
- Environment-driven configuration

Run with: python -m rlvr.train
"""

import ctypes
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List

from rlvr.environment import compute_reward_with_metadata, prepend_signature
from rlvr.logging import RewardLogger, log_learning_metrics


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

DEFAULT_POOL_SIZE = 4


# ============================================================================
# TRL Adapter
# ============================================================================


def _score_single(args: tuple) -> Dict[str, Any]:
    """Score a single completion and return full metadata.

    Defined at module level to be picklable for ProcessPoolExecutor.
    """
    pid, completion, tests, raw_completion = args
    info = {"tests": tests, "problem_id": pid, "raw_completion": raw_completion}
    _, metadata = compute_reward_with_metadata(completion, info, {})
    return metadata


def create_reward_function(
    logger: RewardLogger | None = None,
    parallel: bool = True,
    pool_size: int | None = None,
) -> Callable:
    """
    Create a TRL-compatible reward function with detailed logging.

    Args:
        logger: RewardLogger for detailed logging. If None, no logging is performed.
        parallel: Whether to score completions in parallel (default True)
        pool_size: Number of parallel workers (default from REWARD_POOL_SIZE env var or 4)

    Returns:
        Function matching TRL's (prompts, completions, **kwargs) -> List[float]
    """
    actual_pool_size = pool_size or int(os.environ.get("REWARD_POOL_SIZE", DEFAULT_POOL_SIZE))

    def _score_parallel(args_list: List[tuple], pool_size: int) -> List[Dict[str, Any]]:
        """Score completions in parallel using a process pool."""
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            future_to_idx = {
                executor.submit(_score_single, args): idx for idx, args in enumerate(args_list)
            }
            results = [None] * len(args_list)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def reward_func(
        prompts: List[str],
        completions: List[str],
        problem_id: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        """Score completions and return rewards."""
        if not completions:
            return []

        ids = problem_id or kwargs.get("problem_id") or []
        tests_list = kwargs.get("tests") or []

        full_completions = [prepend_signature(p, c) for p, c in zip(prompts, completions)]

        args_list = [
            (
                ids[idx] if idx < len(ids) else f"sample_{idx}",
                full_completions[idx],
                tests_list[idx] if idx < len(tests_list) else "",
                completions[idx],  # raw completion for degenerate detection
            )
            for idx in range(len(full_completions))
        ]

        if parallel and len(args_list) > 1:
            results = _score_parallel(args_list, actual_pool_size)
        else:
            results = [_score_single(args) for args in args_list]

        rewards = []
        detailed_logs = []
        completion_logs = []

        for idx, result in enumerate(results):
            pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
            completion = completions[idx]

            rewards.append(float(result["total_reward"]))
            detailed_logs.append(_build_detailed_log_entry(pid, completion, result))
            completion_logs.append(_build_completion_log_entry(pid, completion, result))

        if logger:
            logger.log("syntax_aware_breakdown", detailed_logs)
            logger.log("completions", completion_logs)

        return rewards

    reward_func.__name__ = "compute_reward"
    return reward_func


def _build_detailed_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build detailed log entry for syntax_aware_breakdown.jsonl."""
    entry = {
        "problem_id": pid,
        "total_reward": float(result["total_reward"]),
        "base_reward": float(result["base_reward"]),
        "type_check": float(result["type_score"]),
        "compile": float(result["compile_score"]),
        "tests": float(result["test_score"]),
        "syntax_errors": result.get("syntax_errors"),
        "error_sample": result.get("error_details"),
        "is_degenerate": result["is_degenerate"],
        "style_penalty": result.get("style_penalty", 0.0),
        "style_reasons": result.get("style_reasons", []),
        "preview": completion[:200],
    }
    if result.get("timeout_stage"):
        entry["timeout_stage"] = result["timeout_stage"]
    return entry


def _build_completion_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build completion log entry for completions.jsonl."""
    entry = {
        "problem_id": pid,
        "reward": float(result["total_reward"]),
        "base_reward": float(result["base_reward"]),
        "length": len(completion),
        "is_degenerate": result["is_degenerate"],
        "style_penalty": result.get("style_penalty", 0.0),
        "completion": completion,
    }
    if result.get("timeout_stage"):
        entry["timeout_stage"] = result["timeout_stage"]
    if result.get("reason"):
        entry["reason"] = result["reason"]
    return entry


# ============================================================================
# Training Entry Point
# ============================================================================


def resolve_model_id() -> str:
    """Return a Hugging Face model identifier suitable for GRPO training."""
    candidate = os.environ.get("BASE_MODEL_ID", "").strip()
    if not candidate:
        raise ValueError("BASE_MODEL_ID environment variable is required")
    return candidate


def main():
    # Import heavy dependencies only when training
    from peft import PeftConfig
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOTrainer

    from rlvr.config import create_grpo_config, create_lora_config
    from rlvr.data import DEFAULT_TRAINING_DATASET, build_training_dataset, create_tokenizer

    class LearningMetricsCallback(TrainerCallback):
        """Callback that logs essential learning metrics using log_learning_metrics."""

        def __init__(self, log_path: Path) -> None:
            self.log_path = log_path

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called when trainer logs metrics."""
            if logs is not None:
                log_learning_metrics(self.log_path, logs)

    training_dataset = os.environ.get("TRAINING_DATASET", DEFAULT_TRAINING_DATASET)
    grpo_output_dir = os.environ.get("GRPO_OUTPUT_DIR", "grpo_runs")

    model_id = resolve_model_id()
    dataset = build_training_dataset(training_dataset)
    tokenizer = create_tokenizer(model_id)

    output_path = Path(grpo_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_logger = RewardLogger(output_path / "reward_logs")

    # Create reward function with detailed logging
    reward_funcs = [create_reward_function(reward_logger)]

    config = create_grpo_config()

    # Check for existing checkpoint to resume from
    last_checkpoint = get_last_checkpoint(grpo_output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        # Load LoRA config from checkpoint to ensure compatibility
        # TODO: Consider creating a new lora config using checkpoint data
        lora_config = PeftConfig.from_pretrained(last_checkpoint)
        lora_config.inference_mode = False
        print(
            f"Loaded LoRA config from checkpoint (r={lora_config.r}, alpha={lora_config.lora_alpha})"
        )
    else:
        print("No checkpoint found. Starting training from scratch.")
        lora_config = create_lora_config()

    # Create learning metrics callback
    metrics_log_path = output_path / "metrics.jsonl"
    learning_callback = LearningMetricsCallback(metrics_log_path)

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[learning_callback],
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(grpo_output_dir)
    tokenizer.save_pretrained(grpo_output_dir)


if __name__ == "__main__":
    main()
