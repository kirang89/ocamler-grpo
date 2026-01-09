---
license: apache-2.0
base_model: {{BASE_MODEL_ID}}
tags:
  - ocaml
  - code
  - {{TRAINING_TYPE}}  # sft or grpo
  - fine-tuned
  - lora
datasets:
  - {{DATASET_ID}}
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# {{MODEL_NAME}}

This model is a fine-tuned version of [{{BASE_MODEL_ID}}](https://huggingface.co/{{BASE_MODEL_ID}}) specialized for generating OCaml code.

## Model Details

- **Base Model:** {{BASE_MODEL_ID}}
- **Fine-tuning Method:** {{TRAINING_METHOD}}  # e.g., "Supervised Fine-Tuning (SFT) with LoRA" or "GRPO with LoRA"
- **Training Dataset:** [{{DATASET_ID}}](https://huggingface.co/datasets/{{DATASET_ID}})
- **LoRA Adapter:** [{{LORA_ADAPTER_ID}}](https://huggingface.co/{{LORA_ADAPTER_ID}})

## Training Configuration

### {{TRAINING_TYPE}} Parameters

| Parameter | Value |
|-----------|-------|
| Batch Size | {{BATCH_SIZE}} |
| Gradient Accumulation Steps | {{GRAD_ACCUM_STEPS}} |
| Effective Batch Size | {{EFFECTIVE_BATCH_SIZE}} |
| Learning Rate | {{LEARNING_RATE}} |
| Number of Epochs | {{NUM_EPOCHS}} |
| Max Sequence Length | {{MAX_SEQ_LENGTH}} |
| LR Scheduler Type | {{LR_SCHEDULER_TYPE}} |
| Warmup Ratio | {{WARMUP_RATIO}} |
| Weight Decay | {{WEIGHT_DECAY}} |
| Max Grad Norm | {{MAX_GRAD_NORM}} |
| Optimizer | {{OPTIMIZER}} |
| Dataloader Num Workers | {{DATALOADER_NUM_WORKERS}} |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | {{LORA_R}} |
| LoRA Alpha | {{LORA_ALPHA}} |
| LoRA Dropout | {{LORA_DROPOUT}} |

### Training Settings

| Parameter | Value |
|-----------|-------|
| Logging Steps | {{LOGGING_STEPS}} |
| Eval Steps | {{EVAL_STEPS}} |
| Save Steps | {{SAVE_STEPS}} |
| Save Total Limit | {{SAVE_TOTAL_LIMIT}} |

<!-- For GRPO models, add this section:
### GRPO-Specific Parameters

| Parameter | Value |
|-----------|-------|
| Num Generations | {{NUM_GENERATIONS}} |
| Temperature | {{TEMPERATURE}} |
| Beta (KL coefficient) | {{BETA}} |
-->

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{{HF_MODEL_ID}}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "user", "content": "Write an OCaml function to compute the factorial of a number."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

This model is designed for generating OCaml code solutions given natural language problem descriptions. It has been fine-tuned on OCaml programming problems to improve its ability to produce correct, idiomatic OCaml code.

## Limitations

- The model may not always produce syntactically correct OCaml code
- Complex algorithmic problems may require multiple attempts
- The model works best with clear, well-specified problem descriptions

## Training Infrastructure

Trained using [TRL](https://github.com/huggingface/trl)'s {{TRAINER_CLASS}} with {{TRAINING_DETAILS}}.

<!--
=== TEMPLATE USAGE ===

Replace the following placeholders:

Required:
- {{BASE_MODEL_ID}}: e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct"
- {{MODEL_NAME}}: e.g., "Qwen2.5-Coder-1.5B-Instruct OCaml SFT"
- {{TRAINING_TYPE}}: "sft" or "grpo"
- {{TRAINING_METHOD}}: e.g., "Supervised Fine-Tuning (SFT) with LoRA"
- {{DATASET_ID}}: e.g., "kiranpg/ocaml-sft-problems"
- {{LORA_ADAPTER_ID}}: HuggingFace ID of the LoRA adapter
- {{HF_MODEL_ID}}: Full HuggingFace model ID for the merged model

Training Parameters:
- {{BATCH_SIZE}}: Per-device batch size
- {{GRAD_ACCUM_STEPS}}: Gradient accumulation steps
- {{EFFECTIVE_BATCH_SIZE}}: BATCH_SIZE * GRAD_ACCUM_STEPS
- {{LEARNING_RATE}}: e.g., "2e-5"
- {{NUM_EPOCHS}}: Number of training epochs
- {{MAX_SEQ_LENGTH}}: Maximum sequence length
- {{LR_SCHEDULER_TYPE}}: e.g., "cosine"
- {{WARMUP_RATIO}}: e.g., "0.01"
- {{WEIGHT_DECAY}}: e.g., "0.05"
- {{MAX_GRAD_NORM}}: e.g., "1.0"
- {{OPTIMIZER}}: e.g., "adamw_8bit"
- {{DATALOADER_NUM_WORKERS}}: e.g., "4"

LoRA Parameters:
- {{LORA_R}}: LoRA rank, e.g., "64"
- {{LORA_ALPHA}}: LoRA alpha, e.g., "128"
- {{LORA_DROPOUT}}: LoRA dropout, e.g., "0.1"

Training Settings:
- {{LOGGING_STEPS}}: e.g., "1"
- {{EVAL_STEPS}}: e.g., "5"
- {{SAVE_STEPS}}: e.g., "10"
- {{SAVE_TOTAL_LIMIT}}: e.g., "5"

Trainer Info:
- {{TRAINER_CLASS}}: "SFTTrainer" or "GRPOTrainer"
- {{TRAINING_DETAILS}}: e.g., "completion-only training (prompts masked from loss)"
-->
