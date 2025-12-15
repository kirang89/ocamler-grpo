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
    Logs essential learning metrics to a dedicated file for easy monitoring.

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
    ]

    # Extract available metrics
    filtered_metrics = {k: v for k, v in metrics.items() if k in essential_keys}

    # Only log if we have meaningful metrics (skip if only epoch or empty)
    if len(filtered_metrics) <= 1:
        return

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize file with header if it doesn't exist
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("GRPO Training - Learning Metrics Log\n")
            f.write("=" * 80 + "\n\n")

    # Format and write log entry
    with log_path.open("a", encoding="utf-8") as f:
        # Epoch indicator
        epoch = filtered_metrics.get("epoch", "?")
        f.write(f"[Epoch {epoch:.2f}]")

        # Core training metrics
        if "loss" in filtered_metrics:
            f.write(f"  loss={filtered_metrics['loss']:.4f}")
        if "grad_norm" in filtered_metrics:
            f.write(f"  grad={filtered_metrics['grad_norm']:.4f}")
        if "learning_rate" in filtered_metrics:
            f.write(f"  lr={filtered_metrics['learning_rate']:.2e}")

        # Reward metrics
        if "reward" in filtered_metrics:
            reward = filtered_metrics["reward"]
            reward_std = filtered_metrics.get("reward_std", 0)
            f.write(f"  reward={reward:.3f}±{reward_std:.3f}")

        if "rewards/syntax_aware_reward/mean" in filtered_metrics:
            rew_mean = filtered_metrics["rewards/syntax_aware_reward/mean"]
            rew_std = filtered_metrics.get("rewards/syntax_aware_reward/std", 0)
            f.write(f"  syntax_rew={rew_mean:.3f}±{rew_std:.3f}")

        # Policy health metrics
        if "entropy" in filtered_metrics:
            f.write(f"  entropy={filtered_metrics['entropy']:.3f}")
        if "frac_reward_zero_std" in filtered_metrics:
            f.write(f"  frac_zero_std={filtered_metrics['frac_reward_zero_std']:.2f}")

        f.write("\n")
