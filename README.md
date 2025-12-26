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

## Post-Training

### Merging the Adapter

After training, merge the LoRA adapter into the base model to create a standalone model:

```bash
uv run scripts/merge_adapter.py <checkpoint-path>
```

This requires `BASE_MODEL_ID` to be set in `.envrc`.

### Converting to GGUF

To convert the merged model to GGUF format for use with llama.cpp:

1. Clone llama.cpp:
   ```bash
   git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
   ```

2. Install dependencies and convert:
   ```bash
   pip install -r llama.cpp/requirements.txt

   python llama.cpp/convert_hf_to_gguf.py merged_model --outfile model.gguf

   # or quantize for smaller size
   llama.cpp/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
   ```

## Evaluate Model Performance

Assess model performance against test cases:

```bash
uv run python evaluate.py
```

## Configuration

All parameters using for training in `train.py` can be configured by environment variables that should be added to `.envrc`.

## Reward System Architecture

The training uses a graduated reward system that provides learning signals at multiple compilation stages:

| Stage | Weight | Description |
|-------|--------|-------------|
| Type Check | 25% | Graduated partial credit based on error count (0 errors = 100%, 1 error = 80%, etc.) |
| Compilation | 10% | Partial credit for successful compilation |
| Tests | 65% | Full credit only when all tests pass |

**Prose Penalty:** Completions detected as degenerate (conversational prose, low keyword density, spam) receive a 0.3x multiplier on their base reward.

The reward system uses the [verifiers](https://github.com/primeintellect-ai/verifiers) library for environment abstraction while maintaining compatibility with trl's GRPOTrainer.

## Metrics

There are three key logs to understand the training:
1. `training.log` -> Logs the training. Good for checking progress. Verbose.
2. `learning.log` -> Logs the specific learning metrics of interest. For more information about them, refer [the doc](doc/metrics.md).
3. `completions.jsonl` -> Structured log of model completions:
   ```json
   {"problem_id": "", "reward": 0.0, "base_reward": 0.0, "length": 895, "prose_penalty_applied": false, "completion": ""}
   ```
4. `syntax_aware_breakdown.jsonl` -> Detailed reward breakdown per completion:
   ```json
   {"problem_id": "", "total_reward": 0.0, "type_check": 0.25, "compile": 0.10, "tests": 0.65, "prose_penalty_applied": false}
   ```

## Project Structure

```
ocamler-grpo/
├── train.py           # Main GRPO training script
├── ocaml_env.py       # Verifiers-compatible environment with reward logic
├── reward_vf.py       # Adapter bridging verifiers env with trl.GRPOTrainer
├── reward.py          # Original reward implementation (kept for reference)
├── logger.py          # Logging infrastructure for training metrics
├── evaluate.py        # Model evaluation script
├── dashboard/         # Real-time training dashboard
│   ├── server.py      # Dashboard backend
│   └── index.html     # Dashboard frontend
├── tests/             # Unit tests
│   └── test_ocaml_env.py
└── scripts/
    └── verify_migration.py  # Compares old vs new reward implementations
```
