# reward.py - TRL-compatible reward function for OCaml code generation
#
# Provides create_reward_function() which returns a reward function compatible
# with trl.GRPOTrainer, with detailed logging for training dashboards.

"""
The trl.GRPOTrainer expects reward functions with signature:
    def reward_func(prompts, completions, problem_id=None, **kwargs) -> List[float]

This module provides create_reward_function() which creates such a function,
using ocaml_reward_with_metadata for scoring and logging detailed breakdowns.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from rlvr.environment import (
    ocaml_reward_with_metadata,
    prepend_signature,
)
from rlvr.logging import RewardLogger

# Default pool size for parallel reward computation
DEFAULT_POOL_SIZE = 4


def create_reward_function(
    logger: RewardLogger | None = None,
    parallel: bool = True,
    pool_size: int | None = None,
) -> Callable:
    """
    Create a TRL-compatible reward function with detailed logging.

    This is the main interface for creating reward functions for GRPO training.
    It uses ocaml_reward_with_metadata for scoring and logs detailed breakdowns
    to syntax_aware_breakdown.jsonl and completions.jsonl.

    Args:
        logger: RewardLogger for detailed logging. If None, no logging is performed.
        parallel: Whether to score completions in parallel (default True)
        pool_size: Number of parallel workers (default from REWARD_POOL_SIZE env var or 4)

    Returns:
        Function matching TRL's (prompts, completions, **kwargs) -> List[float]

    Example:
        >>> from rlvr.reward import create_reward_function
        >>> from rlvr.logging import RewardLogger
        >>> logger = RewardLogger(Path("logs"))
        >>> reward_fn = create_reward_function(logger)
        >>> rewards = reward_fn(prompts, completions, problem_id=ids, tests=tests_list)
    """
    actual_pool_size = pool_size or int(os.environ.get("REWARD_POOL_SIZE", DEFAULT_POOL_SIZE))

    def reward_func(
        prompts: List[str],
        completions: List[str],
        problem_id: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        """
        Score completions and return rewards.

        Args:
            prompts: List of prompts (used for signature extraction)
            completions: List of model completions to score
            problem_id: Optional list of problem IDs for logging
            **kwargs: Must include 'tests' (list of test code strings)

        Returns:
            List of float rewards, one per completion
        """
        if not completions:
            return []

        ids = problem_id or kwargs.get("problem_id") or []
        tests_list = kwargs.get("tests") or []

        # Combine function signatures from prompts with completions
        full_completions = [prepend_signature(p, c) for p, c in zip(prompts, completions)]

        # Build args for scoring
        args_list = [
            (
                ids[idx] if idx < len(ids) else f"sample_{idx}",
                full_completions[idx],
                tests_list[idx] if idx < len(tests_list) else "",
            )
            for idx in range(len(full_completions))
        ]

        # Score completions (parallel or sequential)
        if parallel and len(args_list) > 1:
            results = _score_parallel(args_list, actual_pool_size)
        else:
            results = [_score_single(args) for args in args_list]

        # Extract rewards and build logs
        rewards = []
        detailed_logs = []
        completion_logs = []

        for idx, result in enumerate(results):
            pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
            completion = completions[idx]

            rewards.append(float(result["total_reward"]))
            detailed_logs.append(_build_detailed_log_entry(pid, completion, result))
            completion_logs.append(_build_completion_log_entry(pid, completion, result))

        # Log results
        if logger:
            logger.log("syntax_aware_breakdown", detailed_logs)
            logger.log("completions", completion_logs)

        return rewards

    reward_func.__name__ = "ocaml_reward"
    return reward_func


def _score_single(args: tuple) -> Dict[str, Any]:
    """Score a single completion and return full metadata."""
    pid, completion, tests = args
    info = {"tests": tests, "problem_id": pid}
    _, metadata = ocaml_reward_with_metadata(completion, info, {})

    # Add prose_penalty_applied for backward compatibility (same as is_degenerate)
    metadata["prose_penalty_applied"] = metadata["is_degenerate"]

    return metadata


def _score_parallel(
    args_list: List[tuple],
    pool_size: int,
) -> List[Dict[str, Any]]:
    """Score completions in parallel using a process pool."""
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        future_to_idx = {
            executor.submit(_score_single, args): idx
            for idx, args in enumerate(args_list)
        }
        results = [None] * len(args_list)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results


def _build_detailed_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build detailed log entry for syntax_aware_breakdown.jsonl."""
    entry = {
        "problem_id": pid,
        "total_reward": float(result["total_reward"]),
        "base_reward": float(result["base_reward"]),
        "type_check": float(result["type_score"]),
        "compile": float(result["compile_score"]),
        "tests": float(result["test_score"]),
        "syntax_errors": result.get("syntax_errors"),
        "error_sample": result.get("error_details"),
        "prose_penalty_applied": result["prose_penalty_applied"],
        "is_degenerate": result["is_degenerate"],
        "style_penalty": result.get("style_penalty", 0.0),
        "style_reasons": result.get("style_reasons", []),
        "preview": completion[:200],
    }
    if result.get("timeout_stage"):
        entry["timeout_stage"] = result["timeout_stage"]
    return entry


def _build_completion_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build completion log entry for completions.jsonl."""
    entry = {
        "problem_id": pid,
        "reward": float(result["total_reward"]),
        "base_reward": float(result["base_reward"]),
        "length": len(completion),
        "prose_penalty_applied": result["prose_penalty_applied"],
        "style_penalty": result.get("style_penalty", 0.0),
        "completion": completion,
    }
    if result.get("timeout_stage"):
        entry["timeout_stage"] = result["timeout_stage"]
    if result.get("reason"):
        entry["reason"] = result["reason"]
    return entry


# Exports
__all__ = [
    "create_reward_function",
]
