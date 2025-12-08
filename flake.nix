{
  description =
    "Nix dev shell for ocamler-grpo (llama.cpp + OCaml + uv toolchains)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        lib = pkgs.lib;

        llamaCpp = pkgs.llama-cpp;
        llamaCppCuda = pkgs.llama-cpp.override { cudaSupport = true; };

        huggingfaceCli = pkgs.python312Packages.huggingface-hub;

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
          direnv
          python312  # Provide Python from Nix for compatibility
        ];

        linuxExtras = lib.optionals pkgs.stdenv.isLinux [
          pkgs.curl.dev
          pkgs.util-linux
          pkgs.which
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cuda_cudart
          pkgs.cudaPackages.libcublas
          pkgs.stdenv.cc.cc.lib  # Provides libstdc++.so.6 for Python packages with C++ extensions
        ];

        darwinExtras = lib.optionals pkgs.stdenv.isDarwin [ ];

        mkDevShell = llamaPkg:
          let
            # Create a wrapper for llama-server (Linux only, with CUDA setup)
            llamaServerWrapper = if pkgs.stdenv.isLinux then
              pkgs.writeShellScriptBin "llama-server" ''
                # Create a temporary directory for our libcuda.so.1 symlink
                CUDA_STUB_DIR=$(mktemp -d)
                trap "rm -rf $CUDA_STUB_DIR" EXIT

                # Symlink only libcuda.so.1 from system
                if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
                  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_STUB_DIR/libcuda.so.1"
                  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_STUB_DIR/libcuda.so"
                fi

                # Only Nix libraries + our isolated libcuda.so.1
                export LD_LIBRARY_PATH="${
                  pkgs.lib.makeLibraryPath [
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.libcublas
                    pkgs.cudaPackages.cudatoolkit
                  ]
                }:$CUDA_STUB_DIR"

                # Run the actual llama-server from the llamaPkg
                exec ${llamaPkg}/bin/llama-server "$@"
              ''
            else
              # On Darwin, just use llama-server directly without CUDA wrapper
              pkgs.writeShellScriptBin "llama-server" ''
                exec ${llamaPkg}/bin/llama-server "$@"
              '';
          in pkgs.mkShell {
            packages = commonPackages ++ linuxExtras ++ darwinExtras
              ++ [ llamaServerWrapper ];

            shellHook = ''
              # Use Nix's Python to avoid glibc conflicts
              export UV_PYTHON="${pkgs.python312}/bin/python3.12"

              ${lib.optionalString pkgs.stdenv.isLinux ''
                # Add CUDA and libstdc++ to library path for PyTorch GPU support (Linux only)
                export LD_LIBRARY_PATH="${
                  lib.makeLibraryPath [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.cudatoolkit
                    pkgs.cudaPackages.libcublas
                  ]
                }:''${LD_LIBRARY_PATH:-}"
              ''}

              if test -f .envrc; then
                eval "$(direnv hook bash)"
                direnv reload >/dev/null 2>&1 || true
              fi

              ${autoSyncHook}
            '';
          };

        autoSyncHook = ''
          export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
          export UV_PYTHON_DOWNLOADS=never

          if [ -z "''${UV_AUTO_SYNC_DISABLED:-}" ]; then
            if [ ! -f "$PWD/uv.lock" ]; then
              echo "[nix] Lock file not found. Creating lock file and Python environment via uv sync (first run)..."
              uv sync
            elif [ ! -d "$PWD/.venv" ]; then
              echo "[nix] Creating Python environment via uv sync..."
              uv sync --frozen
            else
              echo "[nix] Refreshing locked Python dependencies via uv sync --frozen..."
              uv sync --frozen
            fi
          else
            echo "[nix] Skipping automatic uv sync because UV_AUTO_SYNC_DISABLED is set."
          fi
        '';
      in {
        devShells.default =
          mkDevShell (if pkgs.stdenv.isLinux then llamaCppCuda else llamaCpp);

        devShells.cuda = mkDevShell llamaCppCuda;
        devShells.cpu = mkDevShell llamaCpp;
      });
}
