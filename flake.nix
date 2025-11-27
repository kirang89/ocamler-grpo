{
  description =
    "Nix dev shell for ocamler-grpo (llama.cpp + OCaml + uv toolchains)";

  inputs = {
    # Pin nixpkgs to an unstable snapshot so Python 3.13, llama.cpp, and uv stay available.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # flake-utils lets us target multiple system types without manual boilerplate.
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      # Cover all machines we actively care about: Intel/ARM Linux + macOS.
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in flake-utils.lib.eachSystem supportedSystems (system:
      let
        # Import nixpkgs for the current platform so each dependency is compiled natively.
        # 'inherit system' passes the current system string (e.g. "x86_64-linux") to nixpkgs.
        pkgs = import nixpkgs {
          inherit system;

          # We set allowUnfree = true because CUDA (needed for NVIDIA GPUs) is proprietary software.
          # Without this, Nix refuses to build packages that depend on CUDA.
          config.allowUnfree = true;
        };

        lib = pkgs.lib;

        # 1. Standard llama.cpp (CPU/Metal).
        # Nix packages often have default configurations.
        llamaCpp = pkgs.llama-cpp;

        # 2. CUDA-enabled llama.cpp (Linux only).
        # The '.override' function allows us to change arguments passed to the package builder.
        # Here, we explicitly tell the llama-cpp expression to enable cudaSupport.
        llamaCppCuda = pkgs.llama-cpp.override { cudaSupport = true; };

        # huggingface-cli is distributed via the huggingface-hub Python package.
        huggingfaceCli = pkgs.python3Packages.huggingface-hub;

        # Common toolchain pieces shared across macOS and Linux.
        commonPackages = with pkgs; [
          cmake
          pkg-config
          curl
          openssl
          git
          git-lfs
          opam
          ocaml
          ocamlPackages.findlib
          uv
          huggingfaceCli
          direnv # Added for managing project-specific environment variables
        ];

        # Linux wants the libcurl headers + misc build utils to mimic libcurl4-openssl-dev.
        linuxExtras = lib.optionals pkgs.stdenv.isLinux [
          pkgs.curl.dev
          pkgs.util-linux
          pkgs.which
        ];

        # macOS shells expose Accelerate + Metal explicitly so llama.cpp can link cleanly.
        darwinExtras = lib.optionals pkgs.stdenv.isDarwin [
          # pkgs.darwin.apple_sdk_12_3.frameworks.Accelerate
          # pkgs.darwin.apple_sdk_12_3.frameworks.Metal
        ];

        # Helper function to build a shell with a specific llama.cpp version.
        # This avoids code duplication between the default (CPU) and CUDA shells.
        mkDevShell = llamaPkg:
          pkgs.mkShell {
            # The packages available in the shell environment.
            packages = commonPackages ++ linuxExtras ++ darwinExtras
              ++ [ llamaPkg ];

            # shellHook is a script that runs every time you enter the environment.
            shellHook = ''
              # Basic direnv integration: assume .envrc is allowed and reload it.
              # This ensures that project-specific environment variables defined in .envrc
              # are loaded when you enter the Nix shell.
              if test -f .envrc; then
                eval "$(direnv hook bash)" # Source the direnv hook for bash (common shell)
                direnv reload >/dev/null 2>&1 || true # Reload .envrc, suppressing output on success or error
              fi

              ${autoSyncHook}
            '';
          };

        # Automatically hydrate the Python virtual environment to honor uv.lock.
        autoSyncHook = ''
          export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
          export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
          export UV_PYTHON_DOWNLOADS=auto

          if [ -z "''${UV_AUTO_SYNC_DISABLED:-}" ]; then
            if [ ! -d "$PWD/.venv" ]; then
              echo "[nix] Creating Python environment via uv sync (first run)..."
              uv sync
            else
              echo "[nix] Refreshing locked Python dependencies via uv sync --frozen..."
              uv sync --frozen
            fi
          else
            echo "[nix] Skipping automatic uv sync because UV_AUTO_SYNC_DISABLED is set."
          fi
        '';
      in {
        # The default shell (run with `nix develop`).
        # Uses the standard llama.cpp (CPU or Metal on macOS).
        devShells.default = mkDevShell llamaCpp;

        # The CUDA shell (run with `nix develop .#cuda`).
        # Uses the CUDA-enabled llama.cpp.
        devShells.cuda = mkDevShell llamaCppCuda;
      });
}
