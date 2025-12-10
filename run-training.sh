#!/usr/bin/env bash
# Wrapper script to run training in daemon mode

exec nohup uv run train.py >training.log 2>&1 &
