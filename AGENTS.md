# Repository Guidelines

## Project Structure & Module Organization
The repo is script-driven: `train.py` now encapsulates the GRPO + LoRA trainer, `evaluate.py` performs inference plus OCaml evaluation, and helper utilities (conversion, dataset prep such as `fetch_acecode.py`) live beside them. Source CSVs (`problems10k.csv`, `problems1k.test.csv`, `problems500.test.csv`) and run artifacts (`metrics.csv`, `grpo_runs/`, `generated_ocaml/`) stay at the top level so experiments remain reproducible. Older workflows are preserved under `backup/`, which still houses `run_ocaml_checks.py` for CSV spot checks.

## Build, Test, and Development Commands
- `uv sync` – install the locked Python 3.13 environment (pip users can fall back to `pip install -e .`).
- `uv run python train.py` – run GRPO + LoRA training with the current dataset and env-configured model id.
- `uv run python evaluate.py` – stream problems from the configured CSV (defaults to `problems1k.test.csv`), call Ollama, and emit timestamped CSV plus OCaml files plus logs/matrices.
- `uv run python fetch_acecode.py --rows 10000 --output problems10k.csv` – export prompts from AceCode without downloading the full dataset.
Ensure `ocaml`, `ocamlc`, and `ollama` exist on PATH before invoking any script.

## Coding Style & Naming Conventions
Follow the repo defaults of 4-space indents, generous type hints, and <=100-character lines (enforced by `pyproject.toml`). Use snake_case for Python identifiers, reserve UPPER_SNAKE for configuration constants such as `GRPO_OUTPUT_DIR`, and keep module-level values immutable where possible. Execute `uv run ruff check .` prior to sending a PR to satisfy style, import order, and lint gates.

## Testing Guidelines
Validation hinges on the OCaml toolchain rather than pytest. When touching reward logic, dataset assembly, or subprocess helpers, re-run `uv run python evaluate.py` and inspect both the console PASS/FAIL summary and the newest `metrics.csv` row. For curated CSVs, `uv run python backup/run_ocaml_checks.py` ingests `blanks.csv`; note any remaining failures inside your PR description.

## Commit & Pull Request Guidelines
Keep commit subjects short, imperative, and under ~72 characters (the lone existing commit, “First commit,” sets the precedent). Pull requests should describe the motivation, list changed datasets/models/artifacts, enumerate env vars required to reproduce, and attach the verification command output or CSV snippet. Rebase onto `main`, avoid committing heavyweight `.gguf` artifacts, and document any manual OCaml edits that accompany generator changes.

## Environment & Configuration Tips
Training reads `TRAINING_PROBLEMS_FILE`, `GRPO_MODEL_ID`/`HF_MODEL_ID`, `GRPO_BATCH_SIZE`, `GRPO_NUM_GENERATIONS`, `GRPO_MAX_PROMPT`, `GRPO_OUTPUT_DIR`, and the optional `LORA_*` knobs. Solution sampling depends on `OLLAMA_URL` and `OLLAMA_MODEL`. Keep these in a local env file or shell profile, never in source, and confirm that new output folders stay gitignored before publishing logs or checkpoints.
