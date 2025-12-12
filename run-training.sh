#!/usr/bin/env bash
# Wrapper script to run training and dashboard in daemon mode

nohup uv run train.py >training.log 2>&1 &
nohup uv run python dashboard/server.py >dashboard.log 2>&1 &
