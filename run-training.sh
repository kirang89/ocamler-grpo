#!/usr/bin/env bash
# Wrapper script to run training with CUDA support
# This adds the system NVIDIA driver path only for the training process

set -e

# Find system libcuda.so.1
CUDA_DRIVER_PATH=""
if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    CUDA_DRIVER_PATH="/usr/lib/x86_64-linux-gnu"
elif [ -f /usr/lib64/libcuda.so.1 ]; then
    CUDA_DRIVER_PATH="/usr/lib64"
else
    echo "Warning: Could not find libcuda.so.1 in standard locations"
    echo "Training may run on CPU instead of GPU"
fi

# Add CUDA driver path if found
if [ -n "$CUDA_DRIVER_PATH" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_DRIVER_PATH"
fi

# Run training with uv
exec uv run train.py "$@"
