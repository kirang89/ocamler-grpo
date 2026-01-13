#!/usr/bin/env bash
# Wrapper script to run training and dashboard in daemon mode

nohup uv run python -m rlvr.train >training.log 2>&1 &
nohup uv run python -m dashboard.server >dashboard.log 2>&1 &
