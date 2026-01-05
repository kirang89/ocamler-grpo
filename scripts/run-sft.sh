#!/usr/bin/env bash
#
# Run SFT training for OCaml code generation
#
# Prerequisites:
# 1. Enter nix shell: nix develop .#cuda (or nix develop on Mac)
# 2. Install deps: uv sync --extra cuda
# 3. Set environment variables in .envrc (or export manually)
#
# Usage (from project root):
#   ./scripts/run-sft.sh                       # Full training
#   SFT_NUM_EPOCHS=0.001 ./scripts/run-sft.sh  # Quick sanity check
#

set -e

# Change to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Ensure BASE_MODEL_ID is set
if [ -z "$BASE_MODEL_ID" ]; then
    echo "Error: BASE_MODEL_ID not set."
    echo "Set it in .envrc or export it:"
    echo "  export BASE_MODEL_ID=\"Qwen/Qwen2.5-Coder-1.5B-Instruct\""
    exit 1
fi

# Print configuration
echo "=========================================="
echo "SFT Training Configuration"
echo "=========================================="
echo "Model:     ${BASE_MODEL_ID}"
echo "Dataset:   ${SFT_DATASET:-kiranpg/ocaml-sft-problems}"
echo "Output:    ${SFT_OUTPUT_DIR:-sft_runs}"
echo "Epochs:    ${SFT_NUM_EPOCHS:-3}"
echo "Batch:     ${SFT_BATCH_SIZE:-4}"
echo "Grad Acc:  ${SFT_GRAD_ACCUM_STEPS:-4}"
echo "LR:        ${SFT_LEARNING_RATE:-2e-5}"
echo "LoRA r:    ${LORA_R:-32}"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "${SFT_OUTPUT_DIR:-sft_runs}"

# Run training in background
echo "Starting SFT training..."
nohup uv run python -m sft.train >sft_training.log 2>&1 &

echo "Starting dashboard server..."
nohup uv run python dashboard/server.py >dashboard.log 2>&1 &
