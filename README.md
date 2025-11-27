## Nix-based development shell

This repository now ships a flake-powered dev shell that installs everything the GRPO + OCaml
workflow expects on both Linux and macOS (Intel + Apple Silicon). Highlights:

- `llama.cpp` built for your CPU/GPU target
- `opam`, OCaml, and `uv` for the Python 3.13 GRPO stack
- `cmake`, `pkg-config`, and the equivalent of `libcurl4-openssl-dev` so llama.cpp builds on Linux
- helper CLIs like `huggingface-cli`, `git-lfs`, and `uv`
- automatic `uv sync` whenever you enter the shell (can be disabled)

### 1. Enable flake support (one-time)

```bash
mkdir -p ~/.config/nix
printf "experimental-features = nix-command flakes\n" >> ~/.config/nix/nix.conf
```

If you already manage macOS with [nix-darwin](https://github.com/LnL7/nix-darwin) or Linux via
`/etc/nix/nix.conf`, add the `experimental-features` line there instead.

### 2. Enter the shell

```bash
nix --extra-experimental-features nix-command --extra-experimental-features flakes develop
```

What happens on entry:

1. All requested toolchains (llama.cpp, OCaml, opam, uv, cmake, libcurl headers, huggingface-cli,
   git-lfs, etc.) are added to `PATH`, compiled for your host architecture.
2. `uv sync` runs automatically the first time to hydrate `.venv` from `uv.lock`. Subsequent shells
   run `uv sync --frozen` to ensure the lockfile is honored. Set `UV_AUTO_SYNC_DISABLED=1` before
   `nix develop` if you want to manage the environment manually.

### macOS tips

- Apple Silicon works out of the box because the flake targets both `x86_64-darwin` and
  `aarch64-darwin`. `llama.cpp` links against the Xcode-provided Metal + Accelerate frameworks.
- If you prefer declarative setup, import this flake into your nix-darwin configuration and add the
  dev shell to your `environment.systemPackages`.

### Linux tips

- The shell bundles `curl.dev`, `openssl`, and `pkg-config`, mirroring the `libcurl4-openssl-dev`
  experience from Debian/Ubuntu so llama.cpp can be rebuilt locally if desired.
- `uv sync` pins CPython and all project dependencies according to `uv.lock`, so you can hop straight
  into `uv run python train.py` or `uv run python evaluate.py`.

### Verification checklist

After `nix develop` completes you should be able to run:

```bash
llama-cli --help              # llama.cpp frontend
opam --version                # OCaml package manager
uv run python evaluate.py     # full GRPO toolchain using the managed virtualenv
huggingface-cli --help        # dataset/model utilities
```

If any of these fail, double-check that you enabled flakes and that `uv sync` succeeded on entry.
