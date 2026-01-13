UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  NIX_CMD := nix develop --impure .#cuda
else
  NIX_CMD := nix develop
endif

.PHONY: shell rlvr-train sft-train eval dashboard grpo-train test lint fmt deps help

shell:
	$(NIX_CMD)

rlvr-train:
	./scripts/run-rlvr-training.py

sft-train:
	./scripts/run-sft.sh

eval:
	./scripts/run-eval.sh

dashboard:
	uv run python dashboard/server.py

grpo-train:
	uv run python train.py

test:
	uv run pytest -v

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

deps:
	uv sync

help:
	@printf "Targets:\n"
	@printf "  shell         enter nix shell (cuda on Linux)\n"
	@printf "  deps          install dependencies via uv sync\n"
	@printf "  rlvr-train    run RLVR training wrapper\n"
	@printf "  sft-train     run SFT training wrapper\n"
	@printf "  grpo-train    run GRPO training (train.py)\n"
	@printf "  eval          start eval pipeline\n"
	@printf "  dashboard     run dashboard server\n"
	@printf "  test          run test suite with pytest\n"
	@printf "  lint          run ruff check\n"
	@printf "  fmt           run ruff format\n"
