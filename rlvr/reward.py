# reward.py - Pure reward computation for OCaml code generation
#
# This module contains:
# - Low-level reward functions that operate on file paths
# - Degenerate output detection
# - Style penalty computation

"""
Pure reward computation for OCaml GRPO training.

The low-level reward functions (type_check_reward, compile_reward, tests_reward)
take file paths as input and return RewardResult objects.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# ============================================================================
# Constants
# ============================================================================

MIN_NON_EMPTY_LINES = 2
TYPE_CHECK_TIMEOUT = 5
COMPILE_TIMEOUT = 10
TEST_TIMEOUT = 30
SYNTAX_ERROR_RE = (
    r"Syntax error|Illegal character|unexpected token|Unterminated|This '.*' might be unmatched"
)

# Graduated type error scores
TYPE_ERROR_SCORE_MAP = {
    0: 0.0,
    1: 0.20,
    2: 0.15,
    3: 0.10,
    4: 0.05,
    5: 0.04,
    6: 0.03,
    7: 0.025,
    8: 0.02,
    9: 0.015,
}

# Reward score constants
TYPE_CHECK_MAX_SCORE = 0.25
COMPILE_SUCCESS_SCORE = 0.10
TESTS_PASS_SCORE = 0.65
STYLE_PENALTY_MAX = 0.10
STYLE_PENALTY_EXTRA_CODE_BLOCK = 0.02
STYLE_PENALTY_TRAILING_PROSE = 0.03
TRAILING_PROSE_MIN_LENGTH = 30

# Default pool size for parallel reward computation
DEFAULT_POOL_SIZE = 4


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class RewardResult:
    """Standardized return type for all reward functions."""

    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Utility Functions
# ============================================================================


def count_non_empty_code_lines(code: str) -> int:
    """
    Return non-empty, non-comment OCaml line count for heuristics.

    Args:
        code: OCaml code string

    Returns:
        Number of non-empty, non-comment lines
    """
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("(*"):
            continue
        count += 1
    return count


# ============================================================================
# Low-Level Reward Functions (operate on file paths)
# ============================================================================


def type_check_reward(source_path: Path, workdir: Path) -> RewardResult:
    """
    Run OCaml type check and return graduated score based on error count.

    Args:
        source_path: Path to the .ml source file
        workdir: Working directory for compilation

    Returns:
        RewardResult with score and metadata
    """
    try:
        subprocess.run(
            ["ocamlc", "-c", source_path.name],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
            timeout=TYPE_CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return RewardResult(
            score=0.0,
            metadata={
                "syntax_errors": None,
                "error_details": "[TimeoutExpired] Type check failed",
                "has_syntax_error": False,
                "timed_out": True,
            },
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        has_syntax_error = bool(re.search(SYNTAX_ERROR_RE, stderr, re.IGNORECASE))
        error_count = len(re.findall(r"\bError:", stderr))
        score = 0.0
        msg = ""

        if has_syntax_error:
            error_count = 0
            msg = f"[SYNTAX ERROR] {stderr[:300]}"
        else:
            msg = f"[TYPE ERRORS: {error_count}] {stderr[:250]}"
            score = TYPE_ERROR_SCORE_MAP.get(error_count, 0.01)

        return RewardResult(
            score=score,
            metadata={
                "syntax_errors": error_count,
                "error_details": msg,
                "has_syntax_error": has_syntax_error,
                "timed_out": False,
            },
        )
    except Exception as exc:
        return RewardResult(
            score=0.0,
            metadata={
                "syntax_errors": None,
                "error_details": f"[{type(exc).__name__}] Type check failed",
                "has_syntax_error": False,
                "timed_out": True,
            },
        )

    return RewardResult(
        score=TYPE_CHECK_MAX_SCORE,
        metadata={
            "syntax_errors": 0,
            "error_details": "success",
            "has_syntax_error": False,
            "timed_out": False,
        },
    )


def compile_reward(
    source_path: Path, workdir: Path, output_name: str, type_check: RewardResult
) -> RewardResult:
    """
    Run OCaml compilation and return score based on success and type check state.

    Args:
        source_path: Path to the .ml source file
        workdir: Working directory for compilation
        output_name: Name for the output executable
        type_check: RewardResult from type_check_reward()

    Returns:
        RewardResult with score and metadata
    """
    if type_check.metadata.get("timed_out"):
        return RewardResult(score=0.0, metadata={"timed_out": True})

    try:
        result = subprocess.run(
            ["ocamlc", "-o", output_name, source_path.name],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return RewardResult(score=0.0, metadata={"timed_out": True})
    except Exception:
        return RewardResult(score=0.0, metadata={"timed_out": False})

    if result.returncode == 0:
        return RewardResult(score=COMPILE_SUCCESS_SCORE, metadata={"timed_out": False})
    elif type_check.score == TYPE_CHECK_MAX_SCORE:
        return RewardResult(score=0.05, metadata={"timed_out": False})
    elif type_check.metadata.get("has_syntax_error"):
        return RewardResult(score=0.0, metadata={"timed_out": False})
    else:
        return RewardResult(score=0.01, metadata={"timed_out": False})


def tests_reward(workdir: Path, executable_name: str) -> RewardResult:
    """
    Run compiled executable and return test score.

    Args:
        workdir: Working directory containing the executable
        executable_name: Name of the executable file

    Returns:
        RewardResult with score and metadata
    """
    try:
        result = subprocess.run(
            [f"./{executable_name}"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
        )
        return RewardResult(
            score=TESTS_PASS_SCORE if result.returncode == 0 else 0.0,
            metadata={"timed_out": False},
        )
    except subprocess.TimeoutExpired:
        return RewardResult(score=0.0, metadata={"timed_out": True})


# ============================================================================
# Degenerate Output Detection
# ============================================================================


def is_degenerate_output(completion: str, code: str) -> tuple[bool, list[str]]:
    """
    Multi-signal detection for degenerate outputs (prose, gibberish, spam).

    Args:
        completion: Full completion text
        code: Extracted code block

    Returns:
        Tuple of (is_degenerate: bool, reasons: list of detected issues)
    """
    if os.environ.get("GRPO_DISABLE_PROSE_PENALTY") == "true":
        return False, []

    reasons = []

    # Signal 1: Highly repetitive content
    if len(completion) > 100:
        chunks = [completion[i : i + 50] for i in range(0, len(completion) - 50, 25)]
        if chunks:
            unique_chunks = len(set(chunks))
            repetition_ratio = unique_chunks / len(chunks) if chunks else 1.0
            if repetition_ratio < 0.3:
                reasons.append("repetitive content")

    # Signal 2: Low code purity
    if len(code) > 0 and len(completion) > 0:
        code_purity = len(code) / len(completion)
        if code_purity < 0.5:
            reasons.append("low code ratio")

    # Signal 3: Markdown code block spam
    code_block_count = completion.count("```")
    if code_block_count > 4:
        reasons.append("code block spam")

    # Signal 4: Stub solutions
    code_lines = count_non_empty_code_lines(code)
    code_lower = code.lower()

    stub_indicators = [
        "placeholder",
        "replace with",
        "actual logic",
        "not implemented",
        "your code here",
        "implement this",
        "implement the",
    ]

    has_stub_indicator = any(ind in code_lower for ind in stub_indicators)
    has_assert_false = "assert false" in code_lower
    has_failwith = "failwith" in code_lower
    has_raise_failure = "raise" in code_lower and "failure" in code_lower

    if code_lines < 5 and has_stub_indicator:
        reasons.append("stub solution (indicator in comment)")
    if code_lines < 5 and has_assert_false:
        reasons.append("stub solution (assert false)")
    if code_lines < 5 and (has_failwith or has_raise_failure) and has_stub_indicator:
        reasons.append("stub solution (exception placeholder)")
    if code_lines < 3 and has_failwith:
        reasons.append("stub solution (short failwith)")

    return len(reasons) >= 1, reasons


def compute_solution_style_penalty(
    completion: str, code: str, code_block_re: re.Pattern
) -> tuple[float, list[str]]:
    """
    Compute small penalty for verbose but correct solutions.

    Args:
        completion: Full completion text
        code: Extracted code block
        code_block_re: Compiled regex for matching code blocks

    Returns:
        Tuple of (penalty: 0.0-0.10, reasons: list of detected issues)
    """
    reasons = []
    penalty = 0.0

    # Check 1: Multiple code blocks
    code_block_count = len(code_block_re.findall(completion))
    if code_block_count > 1:
        extra_blocks = code_block_count - 1
        penalty += STYLE_PENALTY_EXTRA_CODE_BLOCK * extra_blocks
        reasons.append(f"{code_block_count} code blocks")

    # Check 2: Trailing prose after final code block
    last_fence = completion.rfind("```")
    if last_fence != -1:
        after_code = completion[last_fence + 3 :].strip()
        if len(after_code) > TRAILING_PROSE_MIN_LENGTH:
            penalty += STYLE_PENALTY_TRAILING_PROSE
            reasons.append("trailing prose")

    penalty = min(penalty, STYLE_PENALTY_MAX)
    return penalty, reasons


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data types
    "RewardResult",
    # Constants
    "MIN_NON_EMPTY_LINES",
    "TYPE_CHECK_MAX_SCORE",
    "COMPILE_SUCCESS_SCORE",
    "TESTS_PASS_SCORE",
    # Utility functions
    "count_non_empty_code_lines",
    # Low-level reward functions
    "type_check_reward",
    "compile_reward",
    "tests_reward",
    # Degenerate and style detection
    "is_degenerate_output",
    "compute_solution_style_penalty",
]
