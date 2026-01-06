#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for OCaml code generation.

Uses TRL's SFTTrainer for efficient training with automatic:
- Prompt masking (loss only on completions)
- Tokenization and padding
- LoRA integration
- Sequence packing (optional)
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from sft.config import load_lora_config_from_env
from sft.data import load_hf_dataset
from sft.logging import (
    format_metrics_jsonl,
    format_metrics_log_line,
    format_train_complete_line,
    format_train_end_record,
    format_train_start_record,
    write_jsonl_record,
    write_log_line,
)


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM_STEPS = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 5
DEFAULT_LR_SCHEDULER_TYPE = "cosine"
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_OPTIMIZER = "adamw_8bit"
DEFAULT_DATALOADER_NUM_WORKERS = 2
DEFAULT_SFT_DATASET = "kiranpg/ocaml-sft-problems"
DEFAULT_SFT_OUTPUT_DIR = "sft_runs"
DEFAULT_EVAL_SPLIT = 0.1
DEFAULT_EVAL_STEPS = 20


# =============================================================================
# Callbacks
# =============================================================================


class SFTMetricsCallback(TrainerCallback):
    """Callback that logs SFT training metrics to files."""

    def __init__(self, log_dir: Path, name: str = "metrics") -> None:
        self.jsonl_path = log_dir / f"{name}.jsonl"
        self.log_path = log_dir / f"{name}.log"
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        write_jsonl_record(self.jsonl_path, format_metrics_jsonl(logs, state.global_step, timestamp))
        write_log_line(self.log_path, format_metrics_log_line(logs))

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        write_jsonl_record(self.jsonl_path, format_train_start_record(
            timestamp=timestamp,
            total_steps=state.max_steps,
            num_epochs=args.num_train_epochs,
            batch_size=args.per_device_train_batch_size,
            grad_accum_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
        ))

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now(timezone.utc).isoformat()
        write_jsonl_record(self.jsonl_path, format_train_end_record(
            timestamp=timestamp,
            total_steps=state.global_step,
            elapsed_seconds=elapsed,
            batch_size=args.per_device_train_batch_size,
        ))
        write_log_line(self.log_path, format_train_complete_line(elapsed, state.global_step))


# =============================================================================
# Helper Functions
# =============================================================================


def get_optimizer() -> str:
    """Get optimizer, falling back to adamw_torch if bitsandbytes unavailable (Mac)."""
    requested = os.environ.get("SFT_OPTIMIZER", DEFAULT_OPTIMIZER)
    if "8bit" in requested:
        try:
            import bitsandbytes  # noqa: F401
            return requested
        except ImportError:
            return "adamw_torch"
    return requested


def load_sft_config_from_env(use_bf16: bool, use_fp16: bool, has_eval: bool) -> SFTConfig:
    """Load SFT training config from environment variables."""
    output_dir = os.environ.get("SFT_OUTPUT_DIR", DEFAULT_SFT_OUTPUT_DIR)
    push_to_hub = os.environ.get("SFT_PUSH_TO_HUB", "false").lower() == "true"
    eval_steps = int(os.environ.get("SFT_EVAL_STEPS", DEFAULT_EVAL_STEPS))

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=float(os.environ.get("SFT_NUM_EPOCHS", DEFAULT_NUM_EPOCHS)),
        per_device_train_batch_size=int(os.environ.get("SFT_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        gradient_accumulation_steps=int(os.environ.get("SFT_GRAD_ACCUM_STEPS", DEFAULT_GRAD_ACCUM_STEPS)),
        learning_rate=float(os.environ.get("SFT_LEARNING_RATE", DEFAULT_LEARNING_RATE)),
        lr_scheduler_type=os.environ.get("SFT_LR_SCHEDULER_TYPE", DEFAULT_LR_SCHEDULER_TYPE),
        warmup_ratio=float(os.environ.get("SFT_WARMUP_RATIO", DEFAULT_WARMUP_RATIO)),
        weight_decay=float(os.environ.get("SFT_WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY)),
        max_grad_norm=float(os.environ.get("SFT_MAX_GRAD_NORM", DEFAULT_MAX_GRAD_NORM)),
        logging_steps=int(os.environ.get("SFT_LOGGING_STEPS", DEFAULT_LOGGING_STEPS)),
        save_steps=int(os.environ.get("SFT_SAVE_STEPS", DEFAULT_SAVE_STEPS)),
        save_total_limit=int(os.environ.get("SFT_SAVE_TOTAL_LIMIT", DEFAULT_SAVE_TOTAL_LIMIT)),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=torch.cuda.is_available(),  # Disable on Mac MPS
        optim=get_optimizer(),
        dataloader_num_workers=int(os.environ.get("SFT_DATALOADER_NUM_WORKERS", DEFAULT_DATALOADER_NUM_WORKERS)),
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        push_to_hub=push_to_hub,
        hub_model_id=os.environ.get("SFT_HUB_MODEL_ID"),
        hub_strategy="checkpoint" if push_to_hub else "end",
        # SFT-specific
        max_length=int(os.environ.get("SFT_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH)),
        packing=False,
        # Evaluation config
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps if has_eval else None,
        per_device_eval_batch_size=int(os.environ.get("SFT_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main training loop."""
    # Get model and dataset from environment
    model_id = os.environ.get("BASE_MODEL_ID", "").strip()
    if not model_id:
        raise ValueError("BASE_MODEL_ID environment variable is required")
    dataset_id = os.environ.get("SFT_DATASET", DEFAULT_SFT_DATASET)
    eval_split = float(os.environ.get("SFT_EVAL_SPLIT", DEFAULT_EVAL_SPLIT))

    # Detect precision
    cuda_available = torch.cuda.is_available()
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and cuda_available

    # Load dataset first to determine if we have eval set
    print(f"Loading dataset from {dataset_id}...")
    train_dataset, eval_dataset = load_hf_dataset(dataset_id, eval_split=eval_split)
    has_eval = eval_dataset is not None
    print(f"  Train: {len(train_dataset)} examples")
    if has_eval:
        print(f"  Eval:  {len(eval_dataset)} examples ({eval_split*100:.0f}% split)")

    # Load configs
    sft_config = load_sft_config_from_env(use_bf16, use_fp16, has_eval=has_eval)
    lora_config = load_lora_config_from_env()

    # Print configuration
    print()
    print(f"Model:           {model_id}")
    print(f"Dataset:         {dataset_id}")
    print(f"Output:          {sft_config.output_dir}")
    print(f"Max seq length:  {sft_config.max_length}")
    print(f"Batch size:      {sft_config.per_device_train_batch_size}")
    print(f"Grad accum:      {sft_config.gradient_accumulation_steps}")
    print(f"Learning rate:   {sft_config.learning_rate}")
    print(f"Epochs:          {sft_config.num_train_epochs}")
    print(f"LoRA r:          {lora_config.r}")
    print(f"LoRA alpha:      {lora_config.lora_alpha}")
    if has_eval:
        print(f"Eval strategy:   every {sft_config.eval_steps} steps")
    print()
    print(f"CUDA available:  {cuda_available}")
    print(f"Using BF16:      {use_bf16}")
    print(f"Using FP16:      {use_fp16}")
    print()

    # Create output directory
    output_path = Path(sft_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  padding_side: {tokenizer.padding_side}")

    # Create SFTTrainer
    # Uses prompt-completion dataset format for automatic prompt masking
    # (completion_only_loss=True by default in SFTConfig)
    print(f"Loading model {model_id} with LoRA...")
    trainer = SFTTrainer(
        model=model_id,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[SFTMetricsCallback(output_path, name="metrics")],
    )
    trainer.model.print_trainable_parameters()

    # Check for checkpoint
    last_checkpoint = get_last_checkpoint(sft_config.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    # Train
    print("Starting SFT training...")
    print(f"  Checkpoints: {output_path}/checkpoint-*/")
    print(f"  Metrics: {output_path}/metrics.jsonl")
    print(f"  Logs:    {output_path}/metrics.log")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"\nTraining complete!")
    print(f"  Checkpoints: {output_path}/checkpoint-*/")
    print(f"  Metrics: {output_path}/metrics.jsonl")


if __name__ == "__main__":
    main()
