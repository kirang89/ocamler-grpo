# ocamler-grpo

**ocamler-grpo** is a machine learning toolkit for fine-tuning LLMs to generate high-quality OCaml code. It aligns models using Generative Representational Preference Optimization (GRPO) with real-time feedback from the OCaml compiler and test suite.

## Prerequisites

- **Nix** (with flakes enabled): Install it from [here](https://nixos.wiki/wiki/Nix_Installation_Guide).

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ocamler-grpo.git
cd ocamler-grpo
```

### 2. Setup Environment with Nix

Enter a development shell with Python, OCaml, uv, and all tools pre-installed:

**macOS:**
```bash
nix develop
```

**Linux with CUDA:**
```bash
nix develop .#cuda
```

### 2.5 Install pytorch with CUDA support

```bash
uv sync --extra cuda
```

### 3. Start Model Server

The Nix environment includes llama.cpp pre-installed. Start a model server:

```bash
llama-server -hf unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF:F16 -c 4096 -ngl -1
```

## Training

Fine-tune the base model (default: Qwen2.5-Coder) using GRPO:

```bash
./run-training.sh
```

This starts the model training using the [default training dataset](https://huggingface.co/datasets/kiranpg/ocaml-training-problems) in the background and logs to `training.log`.


## Evaluate Model Performance

Assess model performance against test cases:

```bash
uv run python evaluate.py
```

## Configuration

All parameters using for training in `train.py` can be configured by environment variables that should be added to `.envrc`.

## Metrics

There are three key logs to understand the training:
1. `training.log` -> Logs the training. Good for checking progress. Verbose.
2. `learning.log` -> Logs the specific learning metrics of interest. For more information about them, refer [the doc](doc/metrics.md).
3. `completions.jsonl` -> Structure log of model completions in the following format:
   ```{"problem_id": "", "reward": 0.0, "length": 895, "runaway_penalty_applied": true/false, "completion":""}```
