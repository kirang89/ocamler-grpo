# environment.py - Verifiers environment setup for OCaml code generation
#
# This module provides:
# - Code extraction utilities (extract_code_block)
# - Function signature handling (prepend_signature, extract_function_signature)
# - Dataset loading (load_ocaml_dataset)
# - Verifiers environment factory (create_ocaml_env)

"""
Verifiers environment setup for OCaml GRPO training.

This module is a thin wrapper that:
- Provides utilities for extracting and manipulating OCaml code
- Loads and transforms datasets into verifiers format
- Creates verifiers SingleTurnEnv with OCaml reward functions
"""

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from rlvr.reward import (
    COMPILE_SUCCESS_SCORE,
    MIN_NON_EMPTY_LINES,
    TESTS_PASS_SCORE,
    TYPE_CHECK_MAX_SCORE,
    RewardResult,
    compile_reward,
    compute_solution_style_penalty,
    count_non_empty_code_lines,
    is_degenerate_output,
    tests_reward,
    type_check_reward,
)

# ============================================================================
# Constants
# ============================================================================

CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "language:ocaml"}

# Pattern to extract function signature from prompt
FUNC_SIGNATURE_RE = re.compile(
    r"(?:let|and)\s+(?:rec\s+)?(?P<name>\w+).*?=(?=(?:\s|\\n)*$)",
    re.DOTALL,
)


# ============================================================================
# Code Extraction Utilities
# ============================================================================


def extract_code_block(text: str) -> str:
    """
    Strip markdown fences so only runnable OCaml reaches the evaluator.

    Handles code blocks with optional language hints (ocaml, ml, etc.).
    If no code blocks found, returns the text as-is.

    Args:
        text: Raw completion text, potentially with markdown code fences

    Returns:
        Extracted OCaml code without markdown formatting
    """
    matches = CODE_BLOCK_RE.findall(text.strip())
    if matches:
        for block in matches:
            block = block.strip()
            if not block:
                continue
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                if first_line.strip().lower() in LANGUAGE_HINTS:
                    return rest.strip()
            if block.lower() in LANGUAGE_HINTS:
                continue
            return block.strip()
    return text.strip()


def extract_function_signature(prompt: str) -> Tuple[str, str]:
    """
    Extract the function signature and name from an OCaml prompt.

    Prompts typically contain a docstring followed by a function signature:
        (**Compute the factorial...
         * >>> factorial 5
         * 120
        *)
        let rec factorial (n : int) : int =

    Args:
        prompt: The full prompt text containing docstring and signature

    Returns:
        A tuple (signature_line, function_name).
        Example: ("let rec factorial (n : int) : int =", "factorial")
        Returns ("", "") if not found.
    """
    search_text = prompt[-300:]
    match = FUNC_SIGNATURE_RE.search(search_text)
    if match:
        return match.group(0).strip(), match.group("name")
    return "", ""


def prepend_signature(prompt: str, completion: str) -> str:
    """
    Prepend the function signature from the prompt to the completion if needed.

    It extracts the signature from the prompt. If the completion does not start
    with a redefinition of the same function, it prepends the signature.

    Args:
        prompt: The prompt text
        completion: The model completion

    Returns:
        The completion with signature prepended if appropriate.
    """
    sig, name = extract_function_signature(prompt)
    if not sig:
        return completion

    stripped = completion.strip()
    if stripped.startswith("let"):
        pattern = re.compile(rf"let\s+(?:rec\s+)?{re.escape(name)}\b")
        if pattern.match(stripped):
            return completion

    return f"{sig}\n  {completion}"


# ============================================================================
# Reward Orchestration
# ============================================================================


def compute_reward_with_metadata(
    completion: str, info: Dict[str, Any], state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Main verifiers rubric function for OCaml code generation with full metadata.

    Implements a graduated reward structure:
    - Type checking: 25% (graduated partial credit for type errors)
    - Compilation: 10% (partial credit based on type check)
    - Tests: 65% (full reward for passing tests)
    - Prose penalty: 0.3x multiplier if degenerate output detected
    - Style penalty: up to 0.10 for passing solutions with style issues

    Args:
        completion: Model's completion text
        info: Dictionary containing test code and problem metadata
              Expected keys: "tests" (str), "problem_id" (str)
        state: Dictionary containing problem state
               Expected keys: "problem_id" (str)

    Returns:
        Tuple of (score, metadata_dict) where:
        - score: Float reward in range [0, 1]
        - metadata_dict: Dictionary with detailed scoring breakdown
    """
    # Extract problem metadata
    problem_id = info.get("problem_id") or state.get("problem_id", "unknown")
    tests = info.get("tests", "")

    # Extract and validate code
    code = extract_code_block(completion)
    if not code or count_non_empty_code_lines(code) < MIN_NON_EMPTY_LINES:
        return 0.0, {
            "problem_id": problem_id,
            "total_reward": 0.0,
            "base_reward": 0.0,
            "type_score": 0.0,
            "compile_score": 0.0,
            "test_score": 0.0,
            "syntax_errors": None,
            "error_details": None,
            "is_degenerate": False,
            "degenerate_reasons": [],
            "style_penalty": 0.0,
            "style_reasons": [],
            "reason": "empty or too short",
            "timeout_stage": None,
            "tests_passed": False,
        }

    # Combine solution with test code
    combined_code = f"{code.rstrip()}\n\n{tests.strip()}\n"

    with tempfile.TemporaryDirectory(prefix=f"{problem_id}_reward_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        source_path = tmpdir / f"{problem_id}.ml"
        source_path.write_text(combined_code, encoding="utf-8")

        # Run compilation pipeline
        type_check_result = type_check_reward(source_path, tmpdir)
        compile_result = compile_reward(source_path, tmpdir, problem_id, type_check_result)

        # Only run tests if compilation succeeded
        compile_succeeded = compile_result.score == COMPILE_SUCCESS_SCORE
        test_result = tests_reward(tmpdir, problem_id) if compile_succeeded else RewardResult(0.0)

    # Determine timeout stage
    timeout_stage = None
    if type_check_result.metadata.get("timed_out"):
        timeout_stage = "type_check"
    elif test_result.metadata.get("timed_out"):
        timeout_stage = "tests"

    # Calculate base reward
    base_reward = type_check_result.score + compile_result.score + test_result.score

    # Apply degenerate output penalty
    is_degen, degen_reasons = is_degenerate_output(completion, code)
    total_reward = base_reward * 0.3 if is_degen else base_reward

    # Apply style penalty for passing solutions
    style_penalty = 0.0
    style_reasons = []
    if base_reward == 1.0 and not is_degen:
        style_penalty, style_reasons = compute_solution_style_penalty(
            completion, code, CODE_BLOCK_RE
        )
        total_reward = total_reward - style_penalty

    # Build reason for reward < 1
    reason = None
    if total_reward < 1.0:
        if style_reasons:
            reason = "style: " + ", ".join(style_reasons)
        elif degen_reasons:
            reason = ", ".join(degen_reasons)
        elif test_result.score == 0.0 and compile_succeeded:
            reason = "test failure"
        elif compile_result.score < COMPILE_SUCCESS_SCORE:
            if type_check_result.metadata.get("has_syntax_error"):
                reason = "syntax error"
            elif type_check_result.score < TYPE_CHECK_MAX_SCORE:
                reason = "type error"
            else:
                reason = "compile failure"
        elif timeout_stage:
            reason = f"timeout ({timeout_stage})"

    metadata = {
        "problem_id": problem_id,
        "total_reward": float(total_reward),
        "base_reward": float(base_reward),
        "type_score": float(type_check_result.score),
        "compile_score": float(compile_result.score),
        "test_score": float(test_result.score),
        "syntax_errors": type_check_result.metadata.get("syntax_errors"),
        "error_details": type_check_result.metadata.get("error_details"),
        "is_degenerate": is_degen,
        "degenerate_reasons": degen_reasons,
        "style_penalty": float(style_penalty),
        "style_reasons": style_reasons,
        "reason": reason,
        "timeout_stage": timeout_stage,
        "tests_passed": bool(test_result.score >= TESTS_PASS_SCORE),
    }

    return float(total_reward), metadata


def compute_reward(completion: str, info: Dict[str, Any], state: Dict[str, Any]) -> float:
    """
    Main verifiers rubric function for OCaml code generation.

    Convenience wrapper around compute_reward_with_metadata that returns just the score.

    Args:
        completion: Model's completion text
        info: Dictionary containing test code and problem metadata
        state: Dictionary containing problem state

    Returns:
        Float reward in range [0, 1]
    """
    score, _ = compute_reward_with_metadata(completion, info, state)
    return score


# ============================================================================
# Dataset Loading
# ============================================================================


def load_ocaml_dataset(dataset_id: str = "kiranpg/ocaml-training-problems"):
    """
    Load and transform OCaml dataset into verifiers format.

    Transforms dataset columns:
    - Keeps: id, prompt, tests
    - Creates: info dict with tests and problem_id

    Args:
        dataset_id: HuggingFace dataset identifier

    Returns:
        Transformed Dataset compatible with verifiers
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="train")

    def transform_example(example):
        """Transform dataset example to verifiers format."""
        return {
            "prompt": example["prompt"],
            "info": {
                "tests": example.get("tests", ""),
                "problem_id": example.get("id", "unknown"),
            },
        }

    return dataset.map(transform_example)


# ============================================================================
# Environment Factory
# ============================================================================


def create_ocaml_env(
    dataset_id: str = "kiranpg/ocaml-training-problems",
    system_prompt: str = "",
):
    """
    Create a verifiers-compatible environment for OCaml code generation.

    The environment uses:
    - OCaml compilation and testing as the reward signal
    - Graduated rewards for partial progress (type errors, compilation)
    - Degenerate output detection to penalize prose/spam

    Args:
        dataset_id: HuggingFace dataset identifier
        system_prompt: System prompt for the model (default empty)

    Returns:
        SingleTurnEnv configured for OCaml code generation
    """
    import verifiers as vf

    dataset = load_ocaml_dataset(dataset_id)

    env = vf.SingleTurnEnv.create(
        system_prompt=system_prompt,
        rubric=[compute_reward],
        dataset=dataset,
    )

    return env


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "CODE_BLOCK_RE",
    # Code extraction
    "extract_code_block",
    "extract_function_signature",
    "prepend_signature",
    # Reward orchestration
    "compute_reward_with_metadata",
    "compute_reward",
    # Dataset and environment
    "load_ocaml_dataset",
    "create_ocaml_env",
]
