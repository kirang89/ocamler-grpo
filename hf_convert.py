from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter = "grpo_runs/checkpoint-3000/"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")

# Apply LoRA adapter weights
model = PeftModel.from_pretrained(base_model, adapter, torch_dtype="auto")

# Load the base model and immediately apply the LoRA adapter weights.
# model = AutoPeftModelForCausalLM.from_pretrained(base, peft_model_id=adapter, torch_dtype="auto")

# Fold the adapter deltas into the base weights so the result is a standalone HF model.
model = model.merge_and_unload()

# Persist merged weights/config to disk for downstream conversion.
model.save_pretrained("merged_qwen_grpo")

# Copy tokenizer artifacts to the same folder to keep the model package complete.
AutoTokenizer.from_pretrained(base).save_pretrained("merged_qwen_grpo")
