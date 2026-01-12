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
from pathlib import Path

from peft import PeftConfig
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer

from rlvr.config import create_grpo_config, create_lora_config
from rlvr.data import DEFAULT_TRAINING_DATASET, build_training_dataset, create_tokenizer
from rlvr.logging import RewardLogger, log_learning_metrics
from rlvr.reward import build_reward_functions_vf


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

# Environment variables for training
TRAINING_DATASET = os.environ.get("TRAINING_DATASET", DEFAULT_TRAINING_DATASET)
GRPO_OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", "grpo_runs")


class LearningMetricsCallback(TrainerCallback):
    """Callback that logs essential learning metrics using log_learning_metrics."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is not None:
            log_learning_metrics(self.log_path, logs)


def resolve_model_id() -> str:
    """Return a Hugging Face model identifier suitable for GRPO training."""
    candidate = os.environ.get("BASE_MODEL_ID", "").strip()
    if not candidate:
        raise ValueError("BASE_MODEL_ID environment variable is required")
    return candidate


def main():
    model_id = resolve_model_id()
    dataset = build_training_dataset(TRAINING_DATASET)
    tokenizer = create_tokenizer(model_id)

    output_path = Path(GRPO_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_logger = RewardLogger(output_path / "reward_logs")
    # Use verifiers-based reward functions (migration from reward.py)
    reward_funcs = build_reward_functions_vf(TRAINING_DATASET, reward_logger)

    config = create_grpo_config()

    # Check for existing checkpoint to resume from
    last_checkpoint = get_last_checkpoint(GRPO_OUTPUT_DIR)
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

    trainer.save_model(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)


if __name__ == "__main__":
    main()
