#!/usr/bin/env python3
"""
Evaluation script for OCaml code generation using a local LLM.

Reads problems from a CSV file, generates solutions via llama-server API,
evaluates them using the training reward system, and outputs metrics to CSV.
"""

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .constants import FAILURE_STAGE_PATTERNS, PASS_THRESHOLD
from .metrics import compute_failure_stages
from .report import generate_html_report

from rlvr.environment import prepend_signature
from rlvr.reward import _score_completion_vf

# Configuration via environment variables
LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080/v1/chat/completions")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "local-model")
INPUT_CSV = os.environ.get("INPUT_CSV", "promptfoo/tests/ocaml_tests_sample.csv")

SYSTEM_PROMPT = "Respond only with runnable OCaml code (no prose)."

PROMPT_TEMPLATE = """You are an expert OCaml engineer. Complete the following OCaml function.
Respond only with the function implementation (no prose, no markdown).

{problem_text}"""


def call_llama(prompt: str) -> str:
    """
    Call the llama-server API with an OpenAI-compatible request.

    Args:
        prompt: The user prompt to send

    Returns:
        The model's response text

    Raises:
        requests.RequestException: On network errors
        ValueError: On unexpected response format
    """
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(LLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices")
    if not choices:
        raise ValueError("Unexpected response: missing 'choices'")
    message = choices[0].get("message")
    if not message or "content" not in message:
        raise ValueError("Unexpected response: missing 'message.content'")

    return message["content"].strip()


def generate_solution(problem_text: str) -> tuple[str, float]:
    """
    Generate a solution for a problem using the LLM.

    Args:
        problem_text: The problem description with function signature

    Returns:
        Tuple of (completion, generation_time_seconds)
    """
    prompt = PROMPT_TEMPLATE.format(problem_text=problem_text.strip())

    start_time = time.perf_counter()
    completion = call_llama(prompt)
    generation_time = time.perf_counter() - start_time

    return completion, generation_time


def map_reason_to_failure_stage(reason: str | None) -> str:
    """
    Map the reward system's reason to a failure stage.

    Args:
        reason: The reason string from _score_completion_vf

    Returns:
        Failure stage category string
    """
    if reason is None:
        return ""

    reason_lower = reason.lower()

    # Style prefix check
    if reason_lower.startswith("style:"):
        return "style"

    # Exception/fatal error (multi-condition)
    if "exception" in reason_lower or "fatal error" in reason_lower:
        return "execution:exception"

    # Pattern matching
    for pattern, stage in FAILURE_STAGE_PATTERNS:
        if pattern in reason_lower:
            return stage

    return f"other:{reason_lower[:30]}"


def evaluate_solution(pid: str, completion: str, tests: str) -> dict[str, Any]:
    """Evaluate a solution using the reward system."""
    return _score_completion_vf(pid, completion, tests)


def read_problems(input_path: str) -> list[dict[str, Any]]:
    """Read problems from CSV file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _base_scores(eval_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build base score fields from eval result or zeros for errors."""
    if eval_result is None:
        return {
            "total_reward": 0.0,
            "base_reward": 0.0,
            "type_score": 0.0,
            "compile_score": 0.0,
            "test_score": 0.0,
            "failure_stage": "generation_error",
        }
    return {
        "total_reward": eval_result["total_reward"],
        "base_reward": eval_result["base_reward"],
        "type_score": eval_result["type_score"],
        "compile_score": eval_result["compile_score"],
        "test_score": eval_result["test_score"],
        "failure_stage": map_reason_to_failure_stage(eval_result.get("reason")),
    }


def build_result(
    pid: str,
    difficulty: str,
    eval_result: dict[str, Any] | None,
    generation_time: float = 0.0,
    completion: str = "",
) -> dict[str, Any]:
    """Build result dictionary from evaluation or error."""
    return {
        "id": pid,
        "difficulty": difficulty,
        **_base_scores(eval_result),
        "generation_time_sec": round(generation_time, 2),
        "completion_length": len(completion),
    }


def build_completion(
    pid: str,
    difficulty: str,
    problem_text: str,
    tests: str,
    eval_result: dict[str, Any] | None,
    generation_time: float = 0.0,
    completion: str = "",
    full_completion: str = "",
    error: str = "",
) -> dict[str, Any]:
    """Build completion dictionary from evaluation or error."""
    result = {
        "id": pid,
        "difficulty": difficulty,
        "problem_text": problem_text,
        "tests": tests,
        "raw_completion": completion,
        "full_completion": full_completion,
        **_base_scores(eval_result),
        "generation_time_sec": round(generation_time, 2),
        "completion_length": len(completion),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": LLAMA_MODEL,
    }
    if error:
        result["error"] = error
    return result


def process_single_problem(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process a single problem and return (result, completion) dictionaries."""
    pid = problem["id"]
    problem_text = problem["problem"]
    tests = problem["tests"]
    difficulty = problem.get("difficulty", "")

    try:
        completion, generation_time = generate_solution(problem_text)
    except Exception as exc:
        return (
            build_result(pid, difficulty, None),
            build_completion(pid, difficulty, problem_text, tests, None, error=str(exc)),
        )

    full_completion = prepend_signature(problem_text, completion)
    eval_result = evaluate_solution(pid, full_completion, tests)

    return (
        build_result(pid, difficulty, eval_result, generation_time, completion),
        build_completion(
            pid, difficulty, problem_text, tests, eval_result,
            generation_time, completion, full_completion,
        ),
    )


def print_problem_status(i: int, total: int, pid: str, result: dict[str, Any]) -> None:
    """Print processing status for a single problem."""
    print(f"[{i + 1}/{total}] Processing {pid}...", end=" ", flush=True)
    if result["total_reward"] >= PASS_THRESHOLD:
        print(f"PASS (reward={result['total_reward']:.2f})")
    else:
        print(f"FAIL (reward={result['total_reward']:.2f}, stage={result['failure_stage']})")


def process_csv(input_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all problems in the CSV file and return (results, completions)."""
    problems = read_problems(input_path)
    results = []
    completions = []

    for i, problem in enumerate(problems):
        result, completion = process_single_problem(problem)
        results.append(result)
        completions.append(completion)
        print_problem_status(i, len(problems), problem["id"], result)

    return results, completions


RESULT_FIELDNAMES = [
    "id", "difficulty", "total_reward", "base_reward", "type_score",
    "compile_score", "test_score", "failure_stage", "generation_time_sec", "completion_length",
]


def write_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Write results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def write_completions(completions: list[dict[str, Any]], output_path: str) -> None:
    """Write completions to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for completion in completions:
            f.write(json.dumps(completion, ensure_ascii=False) + "\n")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics."""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    passed = sum(1 for r in results if r["total_reward"] >= PASS_THRESHOLD)
    pass_rate = passed / total * 100
    failure_stages = compute_failure_stages(results)

    gen_times = [r["generation_time_sec"] for r in results if r["generation_time_sec"] > 0]
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total problems: {total}")
    print(f"Pass rate: {pass_rate:.1f}% ({passed}/{total})")
    print(f"Average generation time: {avg_gen_time:.2f}s")

    if failure_stages:
        print("\nFailure breakdown:")
        for stage, count in sorted(failure_stages.items(), key=lambda x: -x[1]):
            print(f"  {stage}: {count}")


def main():
    """Main entry point."""
    # Generate output directory with model name and timestamp
    model_name = LLAMA_MODEL.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"eval_runs/{model_name}_{timestamp}")

    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.csv"
    completions_path = run_dir / "completions.jsonl"

    print(f"Input: {INPUT_CSV}")
    print(f"Model: {LLAMA_MODEL}")
    print(f"API URL: {LLAMA_URL}")
    print(f"Output directory: {run_dir}")
    print("-" * 50)

    results, completions = process_csv(INPUT_CSV)
    write_results(results, str(results_path))
    write_completions(completions, str(completions_path))
    generate_html_report(results, run_dir, LLAMA_MODEL, INPUT_CSV)

    print(f"\nResults written to:")
    print(f"  CSV: {results_path}")
    print(f"  Completions: {completions_path}")
    print(f"  Report: {run_dir / 'report.html'}")
    print_summary(results)


if __name__ == "__main__":
    main()
