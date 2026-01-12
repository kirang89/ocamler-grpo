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


# Keys that pass through unchanged from TRL trainer to JSONL
_DIRECT_KEYS = ["epoch", "loss", "grad_norm", "learning_rate", "entropy", "step_time", "kl"]

# Keys that need renaming: TRL trainer key → JSONL key
_RENAMED_KEYS = {
    "reward": "reward_mean",
    "reward_std": "reward_std",
    "rewards/syntax_aware_reward/mean": "syntax_reward_mean",
    "rewards/syntax_aware_reward/std": "syntax_reward_std",
    "frac_reward_zero_std": "frac_zero_std",
    "completions/mean_length": "mean_length",
}

# All keys we track (for filtering)
_ESSENTIAL_KEYS = set(_DIRECT_KEYS) | set(_RENAMED_KEYS.keys())


def format_grpo_metrics_jsonl(metrics: Dict) -> dict | None:
    """
    Format GRPO training metrics as a JSONL record. Pure function.

    Returns None if there are insufficient metrics to log.
    """
    # Extract only the metrics we care about
    filtered = {k: v for k, v in metrics.items() if k in _ESSENTIAL_KEYS}

    # Only log if we have meaningful metrics (skip if only epoch or empty)
    if len(filtered) <= 1:
        return None

    record: Dict[str, Any] = {}

    # Direct keys pass through unchanged
    for key in _DIRECT_KEYS:
        if key in filtered:
            record[key] = filtered[key]

    # Renamed keys get mapped to their JSONL names
    for src, dst in _RENAMED_KEYS.items():
        if src in filtered:
            record[dst] = filtered[src]

    return record


def log_learning_metrics(log_path: Path, metrics: Dict) -> None:
    """
    Logs essential learning metrics to a JSONL file for easy monitoring.

    Args:
        log_path: Path to the metrics.jsonl file
        metrics: Dictionary of training metrics from trainer logs
    """
    record = format_grpo_metrics_jsonl(metrics)
    if record is None:
        return

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append JSON record
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
