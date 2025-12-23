#!/bin/bash
set -eo pipefail

# Get the directory where this script lives (repo/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}[STEP]${NC} $1"
    echo -e "${GREEN}========================================${NC}\n"
}

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root"
    exit 1
fi

# Step 1: Run package update
log_step "Running package update"
apt-get update -y
log_info "Package update completed"

set -eo pipefail

echo "👤 Running as user: $(whoami)"

# Step 7: Install Nix package manager
echo "📦 Installing Nix package manager..."

NIX_INSTALLER_PATH="/tmp/nix-install.sh"

echo "🔗 Resolving Nix installer URL..."
VERSIONED_URL=$(curl --proto '=https' --tlsv1.2 -sSL -o /dev/null -w '%{url_effective}' https://nixos.org/nix/install)
CHECKSUM_URL="${VERSIONED_URL}.sha256"

echo "⬇️  Downloading Nix installer from: $VERSIONED_URL"
curl --proto '=https' --tlsv1.2 -sSL "$VERSIONED_URL" -o "$NIX_INSTALLER_PATH"

echo "⬇️  Downloading official checksum..."
EXPECTED_CHECKSUM=$(curl --proto '=https' --tlsv1.2 -sSL "$CHECKSUM_URL" | tr -d '[:space:]')

echo "🔐 Verifying checksum..."
ACTUAL_CHECKSUM=$(sha256sum "$NIX_INSTALLER_PATH" | cut -d' ' -f1)
if [ "$ACTUAL_CHECKSUM" != "$EXPECTED_CHECKSUM" ]; then
    echo "❌ Nix installer checksum mismatch!"
    echo "   Expected: $EXPECTED_CHECKSUM"
    echo "   Actual:   $ACTUAL_CHECKSUM"
    rm -f "$NIX_INSTALLER_PATH"
    exit 1
fi
echo "✅ Checksum verified"

sudo sh "$NIX_INSTALLER_PATH" --daemon --yes
rm -f "$NIX_INSTALLER_PATH"
echo "✅ Nix installed"

# Source nix profile
if [ -f /etc/profile.d/nix.sh ]; then
    . /etc/profile.d/nix.sh
    echo "✅ Sourced /etc/profile.d/nix.sh"
elif [ -f ~/.nix-profile/etc/profile.d/nix.sh ]; then
    . ~/.nix-profile/etc/profile.d/nix.sh
    echo "✅ Sourced ~/.nix-profile/etc/profile.d/nix.sh"
else
    echo "⚠️  Nix profile not found, you may need to restart your shell"
fi

# Step 8: Create /etc/nix/nix.conf with experimental features
echo "⚙️  Configuring Nix experimental features..."
sudo mkdir -p /etc/nix
echo "experimental-features = nix-command flakes" | sudo tee /etc/nix/nix.conf >/dev/null
echo "✅ Nix flakes enabled"

echo ""
echo "🎉 Bootstrap completed successfully!"

chmod +x "$NIXER_SCRIPT"
chown nixer:nixer "$NIXER_SCRIPT"

log_info "Executing remaining setup as user 'nixer'"
su - nixer -c "$NIXER_SCRIPT"
NIXER_EXIT_CODE=$?
rm -f "$NIXER_SCRIPT"

exit $NIXER_EXIT_CODE
