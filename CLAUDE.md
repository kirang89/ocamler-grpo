# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ocamler-grpo** is a machine learning toolkit for fine-tuning LLMs to generate high-quality OCaml code using Generative Representational Preference Optimization (GRPO) with real-time feedback from the OCaml compiler and test suite.

## Development Environment

**Nix flakes required.** Enter the dev shell before any development work:

```bash
# macOS
nix develop

# Linux with CUDA
nix develop .#cuda
```

The Nix shell provides: Python 3.12, OCaml, uv, llama.cpp, opam, huggingface-cli.

After entering the shell, dependencies auto-sync via `uv sync --frozen`. For CUDA:
```bash
uv sync --extra cuda
```

## Common Commands

```bash
# Run RLVR training (starts in background, logs to training.log)
./run-training.sh

# Run SFT pre-training
./scripts/run-sft.sh

# Quick SFT sanity check
SFT_NUM_EPOCHS=0.001 ./scripts/run-sft.sh

# Run evaluation against test dataset
uv run python -m eval.eval

# Lint check (run before PRs)
uv run ruff check .

# Run tests
uv run pytest

# Merge LoRA adapter after training (requires BASE_MODEL_ID in .envrc)
uv run scripts/merge_adapter.py <checkpoint-path>

# Start local model server for evaluation
llama-server -hf unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF:F16 -c 4096 -ngl -1
```

## Architecture

### Training Pipeline (RLVR)

```
train.py                    # Entry point - configures GRPOTrainer with LoRA
    ├── environment.py      # Verifiers-compatible env with OCaml reward logic
    ├── reward.py           # Adapter bridging verifiers env with trl.GRPOTrainer
    └── logger.py           # Structured logging for training metrics
```

**Reward System** (`environment.py`): Uses a graduated reward structure:
- Type checking: 25% (partial credit scaled by error count, 0 for syntax errors)
- Compilation: 10% (partial credit based on type check)
- Tests: 65% (all-or-nothing for passing)
- Degenerate output penalty: 0.3x multiplier for prose/spam

Reward computation runs in parallel using `ProcessPoolExecutor` for throughput.

### SFT Pipeline

```
sft/
├── train.py              # SFT training with TRL's SFTTrainer + LoRA
├── config.py             # LoRA configuration
├── data.py               # Dataset loading from HuggingFace
└── logging.py            # Metrics logging
```

Uses completion-only training: prompts are masked from loss, model learns only the OCaml code blocks.

### Evaluation

```
eval/
├── eval.py               # Main evaluation runner
├── compare.py            # Compare evaluation runs
├── metrics.py            # Metrics computation
└── report.py             # HTML report generation
```

### Dashboard

```
dashboard/
├── server.py             # Real-time metrics server (port 8080)
└── index.html            # Dashboard frontend
```

## Key Environment Variables

Set in `.envrc` or export manually. **Never commit credentials.**

| Variable | Required | Description |
|----------|----------|-------------|
| `BASE_MODEL_ID` | Yes | HuggingFace model ID for training |
| `TRAINING_DATASET` | No | HF dataset or CSV (default: `kiranpg/ocaml-training-problems`) |
| `GRPO_OUTPUT_DIR` | No | Output directory (default: `grpo_runs`) |
| `GRPO_BATCH_SIZE` | No | Per-device batch size (default: 4) |
| `GRPO_NUM_GENERATIONS` | No | Completions per prompt (default: 4) |
| `GRPO_LEARNING_RATE` | No | Learning rate (default: 5e-6) |
| `LORA_R` | No | LoRA rank (default: 32) |
| `REWARD_POOL_SIZE` | No | Parallel reward workers (default: 4) |
| `OPENAI_BASE_URL` | No | OpenAI-compatible API URL for evaluation |
| `OPENAI_MODEL` | No | Model name for evaluation |

## Coding Conventions

- 4-space indents, <=100 character lines (enforced by `pyproject.toml`)
- Snake_case for Python identifiers
- UPPER_SNAKE for configuration constants
- Type hints on functions
- Run `uv run ruff check .` before submitting PRs

## Training Logs

Monitor training through these files in `GRPO_OUTPUT_DIR`:
- `training.log` - Verbose training output
- `learning.log` - Key learning metrics (reward mean/std, KL divergence)
- `completions.jsonl` - Model completions with rewards
- `syntax_aware_breakdown.jsonl` - Detailed reward breakdown per completion

## Dataset Format

Training datasets (HuggingFace or CSV) require columns:
- `id`: Problem identifier
- `prompt`: Problem description + function signature
- `tests`: OCaml test code to validate solutions

SFT datasets use `prompt` and `solution` columns.
