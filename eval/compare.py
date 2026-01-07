#!/usr/bin/env python3
"""
Compare evaluation results from two model runs.

Usage:
    python eval/compare.py eval_runs/base_model_xxx/results.csv eval_runs/finetuned_xxx/results.csv
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_results(path: str) -> list[dict[str, Any]]:
    """Load results from CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(results: list[dict[str, Any]], name: str = "") -> dict[str, Any]:
    """Compute metrics from results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "name": name}

    # Convert string values to float
    for r in results:
        for key in ["total_reward", "type_score", "compile_score", "test_score"]:
            if key in r and isinstance(r[key], str):
                r[key] = float(r[key])

    passed = sum(1 for r in results if r["total_reward"] >= 1.0)
    type_check_pass = sum(1 for r in results if r["type_score"] >= 0.25)
    compiles = sum(1 for r in results if r["compile_score"] >= 0.10)
    tests_pass = sum(1 for r in results if r["test_score"] >= 0.65)

    # Average scores
    avg_total = sum(r["total_reward"] for r in results) / total

    # Failure stages
    failure_stages: dict[str, int] = {}
    for r in results:
        stage = r.get("failure_stage", "")
        if stage:
            failure_stages[stage] = failure_stages.get(stage, 0) + 1

    return {
        "name": name,
        "total": total,
        "passed": passed,
        "pass_at_1": round(passed / total * 100, 1),
        "type_check_pass": type_check_pass,
        "type_check_rate": round(type_check_pass / total * 100, 1),
        "compiles": compiles,
        "compile_rate": round(compiles / total * 100, 1),
        "tests_pass": tests_pass,
        "test_pass_rate": round(tests_pass / total * 100, 1),
        "avg_reward": round(avg_total, 4),
        "failure_stages": failure_stages,
    }


def compute_per_problem_comparison(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare results problem-by-problem."""
    # Index by problem ID
    a_by_id = {r["id"]: r for r in results_a}
    b_by_id = {r["id"]: r for r in results_b}

    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    improved = []  # B solved, A didn't
    regressed = []  # A solved, B didn't
    both_pass = []
    both_fail = []

    for pid in common_ids:
        a_pass = float(a_by_id[pid]["total_reward"]) >= 1.0
        b_pass = float(b_by_id[pid]["total_reward"]) >= 1.0

        if b_pass and not a_pass:
            improved.append(pid)
        elif a_pass and not b_pass:
            regressed.append(pid)
        elif a_pass and b_pass:
            both_pass.append(pid)
        else:
            both_fail.append(pid)

    return {
        "common_problems": len(common_ids),
        "improved": improved,
        "regressed": regressed,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "net_change": len(improved) - len(regressed),
    }


def format_delta(val_a: float, val_b: float, is_pct: bool = True) -> str:
    """Format the delta between two values with color indicator."""
    delta = val_b - val_a
    if abs(delta) < 0.1:
        indicator = "  "
    elif delta > 0:
        indicator = "↑ "
    else:
        indicator = "↓ "

    if is_pct:
        return f"{indicator}{delta:+.1f}%"
    else:
        return f"{indicator}{delta:+.4f}"


def print_comparison(metrics_a: dict, metrics_b: dict, comparison: dict) -> None:
    """Print side-by-side comparison."""
    name_a = metrics_a["name"] or "Model A"
    name_b = metrics_b["name"] or "Model B"

    # Truncate names for display
    name_a_disp = name_a[:20] if len(name_a) > 20 else name_a
    name_b_disp = name_b[:20] if len(name_b) > 20 else name_b

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"Model A: {name_a}")
    print(f"Model B: {name_b}")
    print(f"Problems: {comparison['common_problems']}")
    print()

    # Main metrics table
    print(f"┌─────────────────────┬────────────┬────────────┬───────────┐")
    print(f"│ Metric              │ {'Model A':^10} │ {'Model B':^10} │ Delta     │")
    print(f"├─────────────────────┼────────────┼────────────┼───────────┤")

    rows = [
        ("Type Check Rate", "type_check_rate", True),
        ("Compile Rate", "compile_rate", True),
        ("Test Pass Rate", "test_pass_rate", True),
        ("pass@1", "pass_at_1", True),
        ("Avg Reward", "avg_reward", False),
    ]

    for label, key, is_pct in rows:
        val_a = metrics_a[key]
        val_b = metrics_b[key]
        delta = format_delta(val_a, val_b, is_pct)

        if is_pct:
            print(f"│ {label:<19} │ {val_a:>8.1f}%  │ {val_b:>8.1f}%  │ {delta:>9} │")
        else:
            print(f"│ {label:<19} │ {val_a:>10.4f} │ {val_b:>10.4f} │ {delta:>9} │")

    print(f"└─────────────────────┴────────────┴────────────┴───────────┘")

    # Problem-level changes
    print(f"\nProblem-Level Changes:")
    print(f"  Improved (B solved, A didn't): {comparison['improved_count']}")
    print(f"  Regressed (A solved, B didn't): {comparison['regressed_count']}")
    print(f"  Both pass: {len(comparison['both_pass'])}")
    print(f"  Both fail: {len(comparison['both_fail'])}")
    print(f"  Net change: {comparison['net_change']:+d}")

    # Show specific problems that changed
    if comparison['improved']:
        print(f"\n  Improved problems: {', '.join(comparison['improved'][:10])}", end="")
        if len(comparison['improved']) > 10:
            print(f" ... (+{len(comparison['improved']) - 10} more)")
        else:
            print()

    if comparison['regressed']:
        print(f"  Regressed problems: {', '.join(comparison['regressed'][:10])}", end="")
        if len(comparison['regressed']) > 10:
            print(f" ... (+{len(comparison['regressed']) - 10} more)")
        else:
            print()

    # Failure stage comparison
    print(f"\nFailure Breakdown:")
    all_stages = set(metrics_a["failure_stages"].keys()) | set(metrics_b["failure_stages"].keys())
    for stage in sorted(all_stages):
        count_a = metrics_a["failure_stages"].get(stage, 0)
        count_b = metrics_b["failure_stages"].get(stage, 0)
        delta = count_b - count_a
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        print(f"  {stage}: {count_a} → {count_b} ({delta_str})")


def write_comparison_json(
    metrics_a: dict,
    metrics_b: dict,
    comparison: dict,
    output_path: str
) -> None:
    """Write comparison to JSON file."""
    data = {
        "model_a": metrics_a,
        "model_b": metrics_b,
        "comparison": comparison,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    if len(sys.argv) < 3:
        print("Usage: python eval/compare.py <results_a.csv> <results_b.csv> [output.json]")
        print("\nExample:")
        print("  python eval/compare.py eval_runs/base/results.csv eval_runs/finetuned/results.csv")
        sys.exit(1)

    path_a = sys.argv[1]
    path_b = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Extract model names from paths
    name_a = Path(path_a).parent.name
    name_b = Path(path_b).parent.name

    print(f"Loading {path_a}...")
    results_a = load_results(path_a)
    print(f"  Loaded {len(results_a)} results")

    print(f"Loading {path_b}...")
    results_b = load_results(path_b)
    print(f"  Loaded {len(results_b)} results")

    metrics_a = compute_metrics(results_a, name_a)
    metrics_b = compute_metrics(results_b, name_b)
    comparison = compute_per_problem_comparison(results_a, results_b)

    print_comparison(metrics_a, metrics_b, comparison)

    if output_path:
        write_comparison_json(metrics_a, metrics_b, comparison, output_path)
        print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
