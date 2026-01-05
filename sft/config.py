"""LoRA configuration for SFT training."""

import os

from peft import LoraConfig, TaskType

# LoRA Defaults
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_BIAS = "none"
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def load_lora_config_from_env() -> LoraConfig:
    """Load LoRA config directly from environment variables."""
    raw_target_modules = os.environ.get("LORA_TARGET_MODULES", DEFAULT_LORA_TARGET_MODULES)
    target_modules = [m.strip() for m in raw_target_modules.split(",") if m.strip()]

    return LoraConfig(
        r=int(os.environ.get("LORA_R", DEFAULT_LORA_R)),
        lora_alpha=int(os.environ.get("LORA_ALPHA", DEFAULT_LORA_ALPHA)),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", DEFAULT_LORA_DROPOUT)),
        bias=os.environ.get("LORA_BIAS", DEFAULT_LORA_BIAS),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
