# ocamler-grpo

**ocamler-grpo** is a machine learning toolkit for fine-tuning LLMs to generate high-quality OCaml code. It aligns models using Generative Representational Preference Optimization (GRPO) with real-time feedback from the OCaml compiler and test suite.

## Prerequisites

- **Python 3.13+**
- **uv** (recommended) or `pip`
- **OCaml** (`ocaml`, `ocamlc` must be in your PATH)
- **Ollama** (required for evaluation inference)

## Setup

You can set up the environment using **Nix** (recommended for reproducibility) or manually using **uv**.

### Option 1: Nix
Run the following to enter a shell with Python, OCaml, and all tools pre-installed:

```bash
nix develop
```

### Option 2: uv
If you are not using Nix, ensure you have **OCaml** installed on your system, then install Python dependencies:

```bash
uv sync
```

## Usage

### 1. Prepare Data
Fetch coding problems from Hugging Face (AceCode-87K):

```bash
uv run python fetch_acecode.py --rows 10000 --output problems10k.csv
```

### 2. Train
Fine-tune the base model (default: Qwen2.5-Coder) using GRPO:

```bash
uv run python train.py
```
*Configuration:* Set `TRAINING_PROBLEMS_FILE` and `GRPO_OUTPUT_DIR` via environment variables to customize inputs/outputs.

### 3. Evaluate
Assess model performance against test cases:

```bash
uv run python evaluate.py
```
*Defaults:* Connects to Ollama at `http://localhost:8080` running `qwen2.5-coder:1.5b-instruct-fp16`.

## Key Files

- `train.py`: Main GRPO training script with custom reward logic (compile/run checks).
- `evaluate.py`: Evaluation harness using Ollama.
- `fetch_acecode.py`: Dataset downloader.
- `GEMINI.md`: Detailed project context and development guide.