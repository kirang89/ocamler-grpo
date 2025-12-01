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

        huggingfaceCli = pkgs.python3Packages.huggingface-hub;

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
        ];

        linuxExtras = lib.optionals pkgs.stdenv.isLinux [
          pkgs.curl.dev
          pkgs.util-linux
          pkgs.which
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cuda_cudart
          pkgs.cudaPackages.libcublas
        ];

        darwinExtras = lib.optionals pkgs.stdenv.isDarwin [ ];

        mkDevShell = llamaPkg:
          let
            # Create a wrapper for llama-server
            llamaServerWrapper = pkgs.writeShellScriptBin "llama-server" ''
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
            '';
          in pkgs.mkShell {
            packages = commonPackages ++ linuxExtras ++ darwinExtras
              ++ [ llamaServerWrapper ];

            shellHook = ''
              if test -f .envrc; then
                eval "$(direnv hook bash)"
                direnv reload >/dev/null 2>&1 || true
              fi

              ${autoSyncHook}
            '';
          };

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
        devShells.default =
          mkDevShell (if pkgs.stdenv.isLinux then llamaCppCuda else llamaCpp);

        devShells.cuda = mkDevShell llamaCppCuda;
        devShells.cpu = mkDevShell llamaCpp;
      });
}
