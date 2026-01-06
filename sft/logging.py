"""
Logging utilities for SFT training.

This module contains:
- Pure functions for formatting metrics
- I/O functions for writing to log files
"""

import json
from pathlib import Path
from typing import Any


# =============================================================================
# Pure Functions: Formatting metrics
# =============================================================================


def format_metrics_log_line(logs: dict) -> str:
    """Format training metrics for human-readable logging. Pure function."""
    epoch = logs.get("epoch", 0)
    loss = logs.get("loss", logs.get("train_loss", 0))
    grad_norm = logs.get("grad_norm", 0)
    lr = logs.get("learning_rate", 0)
    eval_loss = logs.get("eval_loss")

    line = f"[Epoch {epoch:.2f}] loss={loss:.4f}"
    if eval_loss is not None:
        line += f" eval_loss={eval_loss:.4f}"
    line += f" grad={grad_norm:.4f} lr={lr:.2e}\n"
    return line


def format_metrics_jsonl(logs: dict, step: int, timestamp: str) -> dict[str, Any]:
    """
    Format training metrics for JSONL logging. Pure function.

    Includes all available metrics from the Trainer plus metadata.
    """
    # Start with timestamp and step
    record = {
        "timestamp": timestamp,
        "step": step,
    }

    # Core metrics (always present)
    record["epoch"] = logs.get("epoch", 0)
    record["loss"] = logs.get("loss", logs.get("train_loss"))
    record["learning_rate"] = logs.get("learning_rate")
    record["grad_norm"] = logs.get("grad_norm")

    # Optional metrics (may be present depending on trainer config)
    optional_keys = [
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "total_flos",
        "eval_loss",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    ]

    for key in optional_keys:
        if key in logs and logs[key] is not None:
            record[key] = logs[key]

    # Include any other metrics not explicitly handled
    for key, value in logs.items():
        if key not in record and value is not None:
            # Skip non-serializable values
            if isinstance(value, (int, float, str, bool, type(None))):
                record[key] = value

    return record


def format_train_start_record(
    timestamp: str,
    total_steps: int,
    num_epochs: float,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Format training start event record. Pure function."""
    return {
        "timestamp": timestamp,
        "event": "train_start",
        "total_steps": total_steps,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": learning_rate,
    }


def format_train_end_record(
    timestamp: str,
    total_steps: int,
    elapsed_seconds: float,
    batch_size: int,
) -> dict[str, Any]:
    """Format training end event record. Pure function."""
    samples_per_second = total_steps * batch_size / elapsed_seconds if elapsed_seconds > 0 else 0
    return {
        "timestamp": timestamp,
        "event": "train_end",
        "total_steps": total_steps,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": samples_per_second,
    }


def format_train_complete_line(elapsed_seconds: float, total_steps: int) -> str:
    """Format training completion line for human-readable log. Pure function."""
    return f"\nTraining complete in {elapsed_seconds:.1f}s ({total_steps} steps)\n"


# =============================================================================
# I/O Functions: Writing to files
# =============================================================================


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_log_line(path: Path, line: str) -> None:
    """Append a line to a log file."""
    with open(path, "a") as f:
        f.write(line)
