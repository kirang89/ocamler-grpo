#!/bin/bash
set -euo pipefail

MODEL="${OPENAI_MODEL:-kiranpg/qwen2.5-ocamler-sft-model-2026-01-06}"
DATASET="${SFT_DATASET:-kiranpg/ocaml-sft-problems}"
OPENAI_URL="${OPENAI_BASE_URL:-http://localhost:8080/v1/chat/completions}"
WAIT_TIMEOUT="${VLLM_WAIT_TIMEOUT:-300}"
WAIT_INTERVAL=5

SERVER_ORIGIN="http://localhost:8080"
if [[ "${OPENAI_URL}" =~ ^https?://[^/]+ ]]; then
    SERVER_ORIGIN="${BASH_REMATCH[0]}"
fi
MODELS_ENDPOINT="${SERVER_ORIGIN}/v1/models"

echo "Starting vllm server with model: ${MODEL}"
nohup start_vllm_server.sh -m "${MODEL}" >vllm_server.log 2>&1 &

echo "Waiting up to ${WAIT_TIMEOUT}s for vLLM server at ${MODELS_ENDPOINT} to become ready..."
elapsed=0
until curl -s -o /dev/null -m 5 "${MODELS_ENDPOINT}"; do
    if (( elapsed >= WAIT_TIMEOUT )); then
        echo "ERROR: vLLM server did not become ready within ${WAIT_TIMEOUT} seconds." >&2
        exit 1
    fi
    sleep "${WAIT_INTERVAL}"
    elapsed=$((elapsed + WAIT_INTERVAL))
    echo "  still waiting... (${elapsed}s)"
done

echo "vLLM server is ready."
echo "Running evals using model: ${MODEL} on dataset: ${DATASET}"
nohup uv run python -m eval.eval >eval.log 2>&1 &
