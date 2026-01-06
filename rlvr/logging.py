"""Logging utilities for RLVR/GRPO training.

This module provides:
- RewardLogger class for batch reward logging
- log_reward_entries() helper function
- log_learning_metrics() for training progress logging
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class RewardLogger:
    """Writes reward outcomes to JSONL for offline inspection."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, reward_name: str, entries: List[Dict[str, Any]]) -> None:
        path = self.base_dir / f"{reward_name}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs batch-level metrics to a specific file."""
        path = self.base_dir / "batch_metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def log_reward_entries(
    logger: RewardLogger | None,
    reward_name: str,
    ids: List[str],
    completions: List[str],
    rewards: List[float],
) -> None:
    if logger is None:
        return
    entries: List[Dict[str, str]] = []
    for idx, reward in enumerate(rewards):
        completion = completions[idx] if idx < len(completions) else ""
        pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
        entries.append(
            {
                "problem_id": pid,
                "reward": reward,
                "preview": completion[:200],
            }
        )
    logger.log(reward_name, entries)


def log_learning_metrics(log_path: Path, metrics: Dict) -> None:
    """
    Logs essential learning metrics as JSON lines to a dedicated file for easy monitoring.

    Args:
        log_path: Path to the learning.log file
        metrics: Dictionary of training metrics from trainer logs
    """
    # Essential metrics to track
    essential_keys = [
        "epoch",
        "loss",
        "grad_norm",
        "learning_rate",
        "reward",
        "reward_std",
        "rewards/syntax_aware_reward/mean",
        "rewards/syntax_aware_reward/std",
        "entropy",
        "frac_reward_zero_std",
        "step_time",
        "completions/mean_length",
        "kl",
    ]

    # Extract available metrics
    filtered_metrics = {k: v for k, v in metrics.items() if k in essential_keys}

    # Only log if we have meaningful metrics (skip if only epoch or empty)
    if len(filtered_metrics) <= 1:
        return

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build JSON structure with cleaner field names
    log_entry = {}

    if "epoch" in filtered_metrics:
        log_entry["epoch"] = filtered_metrics["epoch"]
    if "loss" in filtered_metrics:
        log_entry["loss"] = filtered_metrics["loss"]
    if "grad_norm" in filtered_metrics:
        log_entry["grad_norm"] = filtered_metrics["grad_norm"]
    if "learning_rate" in filtered_metrics:
        log_entry["learning_rate"] = filtered_metrics["learning_rate"]
    if "reward" in filtered_metrics:
        log_entry["reward_mean"] = filtered_metrics["reward"]
    if "reward_std" in filtered_metrics:
        log_entry["reward_std"] = filtered_metrics["reward_std"]
    if "rewards/syntax_aware_reward/mean" in filtered_metrics:
        log_entry["syntax_reward_mean"] = filtered_metrics["rewards/syntax_aware_reward/mean"]
    if "rewards/syntax_aware_reward/std" in filtered_metrics:
        log_entry["syntax_reward_std"] = filtered_metrics["rewards/syntax_aware_reward/std"]
    if "entropy" in filtered_metrics:
        log_entry["entropy"] = filtered_metrics["entropy"]
    if "frac_reward_zero_std" in filtered_metrics:
        log_entry["frac_reward_zero_std"] = filtered_metrics["frac_reward_zero_std"]
    if "step_time" in filtered_metrics:
        log_entry["step_time"] = filtered_metrics["step_time"]
    if "completions/mean_length" in filtered_metrics:
        log_entry["mean_length"] = filtered_metrics["completions/mean_length"]
    if "kl" in filtered_metrics:
        log_entry["kl"] = filtered_metrics["kl"]

    # Write as single-line JSON
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
