#!/bin/bash
set -eo pipefail

# Step 1: Install Nix package manager
echo "📦 Installing Nix package manager..."
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install)
echo "✅ Nix installed"

# Step 2: Source Nix profile
echo "🔧 Sourcing Nix profile..."
. /home/nixer/.nix-profile/etc/profile.d/nix.sh
# Explicitly add nix to PATH to ensure it's available
export PATH="/home/nixer/.nix-profile/bin:$PATH"
echo "✅ Nix profile sourced"

# Step 3: Configure Nix experimental features
echo "⚙️  Configuring Nix experimental features..."
sudo mkdir -p /etc/nix
echo "experimental-features = nix-command flakes" | sudo tee /etc/nix/nix.conf >/dev/null
echo "✅ Configuration completed"

# Step 4: Link NVIDIA CUDA and NVML libraries
echo "🔗 Linking NVIDIA CUDA and NVML libraries..."
cd /home/nixer/ocamler-grpo
mkdir -p .cuda-driver
cd .cuda-driver
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so .
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 .
sudo ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 .
export LD_LIBRARY_PATH="/home/nixer/ocamler-grpo/.cuda-driver/:$LD_LIBRARY_PATH"
cd ..
echo "✅ CUDA and NVML libraries linked"

# Step 5: Enter nix development shell with CUDA support and run remaining steps
echo "🔧 Entering nix development environment with CUDA support..."
nix develop .#cuda --command bash -c '
set -eo pipefail

# Verify we are inside nix shell
if [ -z "$IN_NIX_SHELL" ]; then
    echo "❌ Error: Not inside nix shell"
    exit 1
fi
echo "✅ Inside nix shell"

# Step 6: Install Python dependencies with CUDA support
echo "📦 Installing Python dependencies with CUDA support..."
uv sync --extra cuda
echo "✅ Python dependencies installed"

# Step 7: Verify PyTorch CUDA support
echo "🔍 Verifying PyTorch CUDA support..."
uv run python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\")"
echo "✅ PyTorch verification complete"
'
echo "✅ Bootstrap complete. Start a new shell with nix develop --impure .#cuda"
