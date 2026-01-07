"""Shared metric computation functions for evaluation."""

from typing import Any

from .constants import (
    COMPILE_THRESHOLD,
    PASS_THRESHOLD,
    TEST_THRESHOLD,
    TYPE_CHECK_THRESHOLD,
)


def compute_metrics(results: list[dict[str, Any]], name: str = "") -> dict[str, Any]:
    """
    Compute evaluation metrics from results.

    Args:
        results: List of result dictionaries
        name: Optional name for the result set

    Returns:
        Dictionary with computed metrics
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "name": name}

    # Normalize string values to float if needed (for CSV loading)
    for r in results:
        for key in ["total_reward", "type_score", "compile_score", "test_score"]:
            if key in r and isinstance(r[key], str):
                r[key] = float(r[key])

    passed = sum(1 for r in results if r["total_reward"] >= PASS_THRESHOLD)
    type_check_pass = sum(1 for r in results if r["type_score"] >= TYPE_CHECK_THRESHOLD)
    compiles = sum(1 for r in results if r["compile_score"] >= COMPILE_THRESHOLD)
    tests_pass = sum(1 for r in results if r["test_score"] >= TEST_THRESHOLD)

    gen_times = [
        r.get("generation_time_sec", 0) for r in results if r.get("generation_time_sec", 0) > 0
    ]

    return {
        "name": name,
        "total": total,
        "passed": passed,
        "type_check_pass": type_check_pass,
        "compiles": compiles,
        "tests_pass": tests_pass,
        "pass_rate": passed / total * 100,
        "pass_at_1": round(passed / total * 100, 1),
        "type_check_rate": type_check_pass / total * 100,
        "compile_rate": compiles / total * 100,
        "test_pass_rate": tests_pass / total * 100,
        "avg_reward": sum(r["total_reward"] for r in results) / total,
        "avg_gen_time": sum(gen_times) / len(gen_times) if gen_times else 0.0,
        "total_gen_time": sum(gen_times),
    }


def compute_failure_stages(results: list[dict[str, Any]]) -> dict[str, int]:
    """
    Compute failure stage breakdown from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping failure stage to count
    """
    failure_stages: dict[str, int] = {}
    for r in results:
        stage = r.get("failure_stage", "")
        if stage:
            failure_stages[stage] = failure_stages.get(stage, 0) + 1
    return failure_stages


def compute_difficulty_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """
    Compute pass/fail breakdown by difficulty level.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping difficulty to {total, passed}
    """
    difficulty_stats: dict[str, dict[str, int]] = {}
    for r in results:
        diff = r.get("difficulty", "unknown") or "unknown"
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "passed": 0}
        difficulty_stats[diff]["total"] += 1
        if r["total_reward"] >= PASS_THRESHOLD:
            difficulty_stats[diff]["passed"] += 1
    return difficulty_stats
