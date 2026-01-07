"""Configuration factories for GRPO training.

This module provides environment-driven configuration for:
- GRPOConfig (TRL trainer configuration)
- LoraConfig (PEFT adapter configuration)
"""

import os

from peft import LoraConfig, TaskType
from trl import GRPOConfig

# GRPO Defaults
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_MAX_PROMPT = 512
DEFAULT_MAX_COMPLETION = 512
DEFAULT_NUM_EPOCHS = 1
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_BETA = 0.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_TOP_ENTROPY_QUANTILE = 0.2
DEFAULT_LOGGING_STEPS = 1
DEFAULT_OUTPUT_DIR = "grpo_runs"
DEFAULT_SAVE_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 30
DEFAULT_TOP_P = 0.95
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_OPTIMIZER = "adamw_8bit"
DEFAULT_DATALOADER_NUM_WORKERS = 2

# LoRA Defaults
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_BIAS = "none"
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def create_grpo_config(temperature=None, output_dir=None) -> GRPOConfig:
    """Assemble GRPO training defaults plus any overrides from env vars.

    Note: These settings have been optimized for running on a RTX 6000 48 GB VRAM.

    Args:
        temperature: Optional sampling temperature override
        output_dir: Optional output directory override

    Returns:
        GRPOConfig configured from environment variables with sensible defaults
    """
    # Deferred import - torch must be imported after _ensure_cuda_driver() in train.py
    import torch

    # Detect CUDA availability
    cuda_available = torch.cuda.is_available()
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    print(f"CUDA support available: {cuda_available}")
    print(f"BF16 support available: {use_bf16}")

    # set to 4 prompts/step if VRAM allows; reduce when using larger models.
    per_device_batch = int(os.environ.get("GRPO_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    # Leave at 1 with batch 4; raise to 2-4 only when you must drop batch size.
    grad_steps = int(os.environ.get("GRPO_GRAD_ACCUM_STEPS", str(DEFAULT_GRAD_ACCUM_STEPS)))
    # Target 4 completions/prompt for the RTX box—turn this up until you near 44 GB VRAM.
    num_generations = int(os.environ.get("GRPO_NUM_GENERATIONS", str(DEFAULT_NUM_GENERATIONS)))
    # Increase to ~512 tokens on the RTX rig to capture full OCaml problems.
    max_prompt = int(os.environ.get("GRPO_MAX_PROMPT", str(DEFAULT_MAX_PROMPT)))
    # Mirror completions at ~512 tokens so solutions + harnesses fit.
    max_completion = int(os.environ.get("GRPO_MAX_COMPLETION", str(DEFAULT_MAX_COMPLETION)))
    # Stick with 1-2 passes; GRPO overfits small OCaml sets quickly.
    num_epochs = float(os.environ.get("GRPO_NUM_EPOCHS", str(DEFAULT_NUM_EPOCHS)))
    # 5e-6 trains safely; bump toward 8e-6 only if the run is stable.
    learning_rate = float(os.environ.get("GRPO_LEARNING_RATE", str(DEFAULT_LEARNING_RATE)))
    # Use KL Coefficient to penalize large policy shifts from reference model
    beta = float(os.environ.get("GRPO_BETA", str(DEFAULT_BETA)))

    generation_batch_size = int(
        os.environ.get("GRPO_GENERATION_BATCH_SIZE", str(per_device_batch * num_generations))
    )

    if temperature is None:
        temperature = float(os.environ.get("GRPO_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

    # Gradient clipping to prevent training instability
    max_grad_norm = float(os.environ.get("GRPO_MAX_GRAD_NORM", str(DEFAULT_MAX_GRAD_NORM)))

    # Optional: Entropy-based token filtering (focuses training on high-entropy tokens)
    # Based on "Beyond the 80/20 Rule" paper - using 20% of highest entropy tokens
    # achieves similar performance to all tokens while improving efficiency
    top_entropy_quantile = float(
        os.environ.get("GRPO_TOP_ENTROPY_QUANTILE", str(DEFAULT_TOP_ENTROPY_QUANTILE))
    )

    # Output directory
    if output_dir is None:
        output_dir = os.environ.get("GRPO_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

    # Checkpointing
    save_steps = int(os.environ.get("GRPO_SAVE_STEPS", str(DEFAULT_SAVE_STEPS)))
    save_total_limit = int(os.environ.get("GRPO_SAVE_TOTAL_LIMIT", str(DEFAULT_SAVE_TOTAL_LIMIT)))

    return GRPOConfig(
        temperature=temperature,  # for training diversity
        top_p=DEFAULT_TOP_P,
        output_dir=output_dir,
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
        logging_steps=int(os.environ.get("GRPO_LOGGING_STEPS", str(DEFAULT_LOGGING_STEPS))),
        bf16=use_bf16,  # Auto-detect bf16 support based on CUDA availability
        gradient_checkpointing=True,
        eval_strategy="no",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        dataloader_num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        beta=beta,
        top_entropy_quantile=top_entropy_quantile,  # Focus training on high-entropy tokens
        lr_scheduler_type="cosine",
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        optim=DEFAULT_OPTIMIZER,
        push_to_hub=True,
        hub_strategy="end",
    )


def create_lora_config() -> LoraConfig:
    """Build a LoraConfig using optional env overrides.

    Returns:
        LoraConfig configured from environment variables with sensible defaults
    """
    # Rank 16 keeps VRAM in check; double it only if you need more adapter capacity.
    lora_r = int(os.environ.get("LORA_R", str(DEFAULT_LORA_R)))
    # Alpha 32 pairs well with r=16; scale roughly 2x the rank when you change it.
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(DEFAULT_LORA_ALPHA)))
    # Small dropout (5%) stabilizes GRPO; set to 0 if you notice underfitting.
    lora_dropout = float(os.environ.get("LORA_DROPOUT", str(DEFAULT_LORA_DROPOUT)))
    # Bias "none" avoids extra params; use "lora_only" when the base model expects it.
    bias = os.environ.get("LORA_BIAS", DEFAULT_LORA_BIAS)
    # Cover attention (q/k/v/o) plus MLP (gate/up/down) blocks for coder backbones.
    raw_target_modules = os.environ.get("LORA_TARGET_MODULES", DEFAULT_LORA_TARGET_MODULES)

    target_modules = [module.strip() for module in raw_target_modules.split(",") if module.strip()]

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
