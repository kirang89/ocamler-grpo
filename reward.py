# reward - Adapter for using verifiers environment with trl.GRPOTrainer
#
# This module provides a bridge between the verifiers SingleTurnEnv
# and trl's reward function interface, preserving existing logging.

"""
The trl.GRPOTrainer expects reward functions with signature:
    def reward_func(prompts, completions, problem_id=None, **kwargs) -> List[float]

The verifiers environment provides:
    ocaml_reward(completion, info, state) -> float

This adapter bridges the two interfaces while maintaining logging compatibility.
"""

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from environment import (
    MIN_NON_EMPTY_LINES,
    RewardResult,
    compile_reward,
    count_non_empty_code_lines,
    extract_code_block,
    is_degenerate_output,
    tests_reward,
    type_check_reward,
)
from logger import RewardLogger

# Default zero-reward result template
REWARDS_ZERO: Dict[str, Any] = {
    "total_reward": 0.0,
    "base_reward": 0.0,
    "type_score": 0.0,
    "compile_score": 0.0,
    "test_score": 0.0,
    "syntax_errors": None,
    "error_details": None,
    "prose_penalty_applied": False,
    "is_degenerate": False,
    "timeout_stage": None,
    "passed": False,
}


def build_reward_functions_vf(dataset_id: str, logger: RewardLogger | None) -> List[Callable]:
    """
    Build reward functions using verifiers environment.

    This replaces the original build_reward_functions() from reward.py,
    providing the same interface but using the verifiers environment internally.

    Args:
        dataset_id: HuggingFace dataset ID or local CSV path
        logger: Optional RewardLogger for detailed logging

    Returns:
        List containing single reward function compatible with trl.GRPOTrainer
    """
    return [make_syntax_aware_reward_vf(logger)]


def make_syntax_aware_reward_vf(logger: RewardLogger | None) -> Callable:
    """
    Create a syntax-aware reward function using verifiers environment components.

    This function maintains the same reward structure as the original but uses
    the verifiers environment's scoring functions directly.

    Args:
        logger: Optional RewardLogger for detailed logging

    Returns:
        Reward function compatible with trl.GRPOTrainer
    """

    def reward_func(
        prompts: List[str],
        completions: List[str],
        completion_ids=None,
        problem_id: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute rewards for completions using verifiers environment.

        Args:
            prompts: List of prompt strings (unused but required by trl)
            completions: List of model completions to score
            completion_ids: Optional completion IDs (unused)
            problem_id: List of problem IDs for logging
            **kwargs: May contain 'tests' key with List[str] of test code

        Returns:
            List of reward scores (floats)
        """
        ids = problem_id or kwargs.get("problem_id") or []
        tests_list = kwargs.get("tests") or []
        rewards = []
        detailed_logs = []
        completion_logs = []

        for idx, completion in enumerate(completions):
            pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
            tests = tests_list[idx] if idx < len(tests_list) else ""

            # Score the completion using verifiers environment logic
            result = _score_completion_vf(pid, completion, tests)

            rewards.append(float(result["total_reward"]))
            detailed_logs.append(_build_detailed_log_entry(pid, completion, result))
            completion_logs.append(_build_completion_log_entry(pid, completion, result))

        # Log results maintaining compatibility with existing dashboard
        if logger:
            logger.log("syntax_aware_breakdown", detailed_logs)
            logger.log("completions", completion_logs)

        return rewards

    reward_func.__name__ = "syntax_aware_reward"
    return reward_func


def _score_completion_vf(
    pid: str, completion: str, tests: str
) -> Dict[str, float | str | bool | None]:
    """
    Score a single completion using verifiers environment scoring logic.

    This function replicates the scoring logic from ocaml_reward() but returns
    detailed results needed for logging (individual score components).

    Args:
        pid: Problem ID
        completion: Model completion text
        tests: Test code to append to completion

    Returns:
        Dictionary containing all scoring details for logging
    """
    # Extract and validate code
    code = extract_code_block(completion)
    if not code or count_non_empty_code_lines(code) < MIN_NON_EMPTY_LINES:
        return {"problem_id": pid, **REWARDS_ZERO}

    # Combine solution with test code
    combined_code = f"{code.rstrip()}\n\n{tests.strip()}\n"

    with tempfile.TemporaryDirectory(prefix=f"{pid}_reward_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        source_path = tmpdir / f"{pid}.ml"
        source_path.write_text(combined_code, encoding="utf-8")

        # Run compilation pipeline
        type_check = type_check_reward(source_path, tmpdir)
        compile_result = compile_reward(source_path, tmpdir, pid, type_check)

        # Only run tests if compilation succeeded
        compile_succeeded = compile_result.score == 0.10
        test_result = tests_reward(tmpdir, pid) if compile_succeeded else RewardResult(0.0)

    # Determine timeout stage
    timeout_stage = None
    if type_check.metadata.get("timed_out"):
        timeout_stage = "type_check"
    elif test_result.metadata.get("timed_out"):
        timeout_stage = "tests"

    # Calculate base reward
    base_reward = type_check.score + compile_result.score + test_result.score

    # Apply degenerate output penalty
    is_degenerate = is_degenerate_output(completion, code)
    total_reward = base_reward * 0.3 if is_degenerate else base_reward

    return {
        "problem_id": pid,
        "total_reward": float(total_reward),
        "base_reward": float(base_reward),
        "type_score": float(type_check.score),
        "compile_score": float(compile_result.score),
        "test_score": float(test_result.score),
        "syntax_errors": type_check.metadata.get("syntax_errors"),
        "error_details": type_check.metadata.get("error_details"),
        "prose_penalty_applied": is_degenerate,
        "is_degenerate": is_degenerate,
        "timeout_stage": timeout_stage,
        "passed": bool(test_result.score >= 0.65),
    }


def _build_detailed_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build detailed log entry for syntax_aware_breakdown.jsonl.

    This format is required for dashboard compatibility.

    Args:
        pid: Problem ID
        completion: Model completion text
        result: Scoring result dictionary from _score_completion_vf()

    Returns:
        Log entry dictionary
    """
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
        "preview": completion[:200],
    }
    if result["timeout_stage"]:
        entry["timeout_stage"] = result["timeout_stage"]
    return entry


def _build_completion_log_entry(
    pid: str, completion: str, result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build completion log entry for completions.jsonl.

    This format stores full completions for analysis.

    Args:
        pid: Problem ID
        completion: Model completion text
        result: Scoring result dictionary from _score_completion_vf()

    Returns:
        Log entry dictionary
    """
    entry = {
        "problem_id": pid,
        "reward": float(result["total_reward"]),
        "base_reward": float(result["base_reward"]),
        "length": len(completion),
        "prose_penalty_applied": result["prose_penalty_applied"],
        "completion": completion,
    }
    if result["timeout_stage"]:
        entry["timeout_stage"] = result["timeout_stage"]
    return entry


# Export main function
__all__ = ["build_reward_functions_vf"]
