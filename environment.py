import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import verifiers as vf
from datasets import Dataset, load_dataset

CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "language:ocaml"}
# Pattern to extract function signature from prompt (the let/let rec ... = part after docstring)
# Handles: literal \n, type definitions between *) and let, trailing whitespace/newlines
FUNC_SIGNATURE_RE = re.compile(
    r"(?:let|and)\s+(?:rec\s+)?(?P<name>\w+).*?=(?=(?:\s|\\n)*$)",
    re.DOTALL,
)
MIN_NON_EMPTY_LINES = 2
TYPE_CHECK_TIMEOUT = 5
COMPILE_TIMEOUT = 10
TEST_TIMEOUT = 30
SYNTAX_ERROR_RE = (
    r"Syntax error|Illegal character|unexpected token|Unterminated|This '.*' might be unmatched"
)
# Graduate type error scores
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


@dataclass
class RewardResult:
    """Standardized return type for all reward functions."""

    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# Code Extraction Utilities


def extract_function_signature(prompt: str) -> Tuple[str, str]:
    """
    Extract the function signature and name from an OCaml prompt.

    Prompts typically contain a docstring followed by a function signature:
        (**Compute the factorial...
         * >>> factorial 5
         * 120
        *)
        let rec factorial (n : int) : int =

    This function extracts the signature line and the function name.

    Args:
        prompt: The full prompt text containing docstring and signature

    Returns:
        A tuple (signature_line, function_name).
        Example: ("let rec factorial (n : int) : int =", "factorial")
        Returns ("", "") if not found.
    """
    # Scan the last 300 characters to be efficient and avoid false positives
    # from the middle of the prompt
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
        # Check for redefinition using regex to handle spacing/rec
        # Regex: let (rec)? name
        pattern = re.compile(rf"let\s+(?:rec\s+)?{re.escape(name)}\b")
        if pattern.match(stripped):
            return completion

    return f"{sig}\n  {completion}"


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


def count_non_empty_code_lines(code: str) -> int:
    """
    Return non-empty, non-comment OCaml line count for heuristics.

    Counts lines that are not empty and do not start with OCaml comment syntax.

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


# Reward Functions


def type_check_reward(source_path: Path, workdir: Path) -> RewardResult:
    """
    Run OCaml type check and return graduated score based on error count.

    Score distribution:
    - 0 errors: 0.25
    - 1 error:  0.20
    - 2 errors: 0.15
    - 3 errors: 0.10
    - 4 errors: 0.05
    - 5 errors: 0.04
    - 6 errors: 0.03
    - 7 errors: 0.025
    - 8 errors: 0.02
    - 9 errors: 0.015
    - 10+ errors: 0.01
    - Syntax errors: 0.0

    Args:
        source_path: Path to the .ml source file
        workdir: Working directory for compilation

    Returns:
        RewardResult with score and metadata containing: syntax_errors,
        error_details, has_syntax_error, timed_out
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
        # raised when return code is non-zero
        stderr = exc.stderr or ""
        has_syntax_error = bool(re.search(SYNTAX_ERROR_RE, stderr, re.IGNORECASE))
        error_count = len(re.findall(r"\bError:", stderr))
        score = 0.0
        msg = ""

        # Syntax errors get zero reward
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

    # Check passed
    return RewardResult(
        score=0.25,
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

    Uses `ocamlc -o` to compile to executable. Provides partial credit based
    on type check results.

    Scoring:
    - Compiles successfully: 0.10 (100% of compile reward)
    - Type checks perfectly but fails to compile: 0.05 (50%)
    - Has type errors and fails to compile: 0.01 (10%)
    - Syntax error: 0.0

    Args:
        source_path: Path to the .ml source file
        workdir: Working directory for compilation
        output_name: Name for the output executable
        type_check: RewardResult from type_check_reward()

    Returns:
        RewardResult with score and metadata containing: timed_out
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

    # Compilation succeeded
    if result.returncode == 0:
        return RewardResult(score=0.10, metadata={"timed_out": False})
    # Perfect type check but compilation failed (linking issues, etc.)
    elif type_check.score == 0.25:
        return RewardResult(score=0.05, metadata={"timed_out": False})
    # Syntax errors get no credit
    elif type_check.metadata.get("has_syntax_error"):
        return RewardResult(score=0.0, metadata={"timed_out": False})
    # Type errors but attempted compilation
    else:
        return RewardResult(score=0.01, metadata={"timed_out": False})


def tests_reward(workdir: Path, executable_name: str) -> RewardResult:
    """
    Run compiled executable and return test score.

    Executes the compiled binary which should contain self-test code.
    Success is determined by zero exit code.

    Args:
        workdir: Working directory containing the executable
        executable_name: Name of the executable file

    Returns:
        RewardResult with score (0.65 if pass, 0.0 otherwise) and metadata
        containing: timed_out
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
            score=0.65 if result.returncode == 0 else 0.0,
            metadata={"timed_out": False},
        )
    except subprocess.TimeoutExpired:
        return RewardResult(score=0.0, metadata={"timed_out": True})


# ============================================================================
# 4. Degenerate Output Detection
# ============================================================================


def is_degenerate_output(completion: str, code: str) -> tuple[bool, list[str]]:
    """
    Multi-signal detection for degenerate outputs (prose, gibberish, spam).

    Returns tuple of (is_degenerate, reasons) where reasons lists detected issues.
    Can be disabled via GRPO_DISABLE_PROSE_PENALTY environment variable.

    This function detects:
    - Repetitive patterns (spam)
    - Low code purity (too much wrapper text)
    - Markdown code block spam (too many ``` markers)
    - Stub solutions (failwith + "implement" in short code)

    Args:
        completion: Full completion text
        code: Extracted code block

    Returns:
        Tuple of (is_degenerate: bool, reasons: list of detected issues)
    """
    if os.environ.get("GRPO_DISABLE_PROSE_PENALTY") == "true":
        return False, []

    reasons = []

    # Signal 1: Highly repetitive content (spam patterns)
    if len(completion) > 100:
        # Check for repeated 50-char chunks
        chunks = [completion[i : i + 50] for i in range(0, len(completion) - 50, 25)]
        if chunks:
            unique_chunks = len(set(chunks))
            repetition_ratio = unique_chunks / len(chunks) if chunks else 1.0

            if repetition_ratio < 0.3:  # >70% repetition
                reasons.append("repetitive content")

    # Signal 2: Low code purity (too much wrapper text)
    if len(code) > 0 and len(completion) > 0:
        code_purity = len(code) / len(completion)

        if code_purity < 0.5:  # Less than half is actual code
            reasons.append("low code ratio")

    # Signal 3: Markdown code block spam (too many ``` markers)
    code_block_count = completion.count("```")
    # Each code block has 2 markers (opening and closing), so count pairs
    # Legitimate completions should have at most 1-2 code blocks (2-4 markers)
    if code_block_count > 4:  # More than 2 code block pairs
        reasons.append("code block spam")

    # Signal 4: Stub solutions (failwith + "implement" in short code)
    # Detects reward hacking like: failwith "Please implement the function."
    # These pass type/compile checks but always fail tests
    code_lines = count_non_empty_code_lines(code)
    code_lower = code.lower()
    if code_lines < 3 and "failwith" in code_lower and "implement" in code_lower:
        reasons.append("stub solution")

    # Require 1+ signals to trigger penalty
    return len(reasons) >= 1, reasons


# ============================================================================
# 5. Reward Functions
# ============================================================================


def ocaml_reward(completion: str, info: Dict[str, Any], state: Dict[str, Any]) -> float:
    """
    Main verifiers rubric function for OCaml code generation.

    Implements a graduated reward structure:
    - Type checking: 25% (graduated partial credit for type errors)
    - Compilation: 10% (partial credit based on type check)
    - Tests: 65% (full reward for passing tests)
    - Prose penalty: 0.3x multiplier if degenerate output detected

    Args:
        completion: Model's completion text
        info: Dictionary containing test code and problem metadata
              Expected keys: "tests" (str), "problem_id" (str)
        state: Dictionary containing problem state
               Expected keys: "problem_id" (str)

    Returns:
        Float reward in range [0, 1]
    """
    # Extract problem metadata
    problem_id = info.get("problem_id") or state.get("problem_id", "unknown")
    tests = info.get("tests", "")

    # Extract and validate code
    code = extract_code_block(completion)
    if not code or count_non_empty_code_lines(code) < MIN_NON_EMPTY_LINES:
        return 0.0

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
        compile_succeeded = compile_result.score == 0.10
        test_result = tests_reward(tmpdir, problem_id) if compile_succeeded else RewardResult(0.0)

    # Calculate base reward
    base_reward = type_check_result.score + compile_result.score + test_result.score

    # Apply degenerate output penalty
    is_degenerate, _ = is_degenerate_output(completion, code)
    total_reward = base_reward * 0.3 if is_degenerate else base_reward

    return float(total_reward)


# ============================================================================
# 6. Dataset Loading
# ============================================================================


def load_ocaml_dataset(dataset_id: str = "kiranpg/ocaml-training-problems") -> Dataset:
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
# 7. Environment Factory
# ============================================================================


def create_ocaml_env(
    dataset_id: str = "kiranpg/ocaml-training-problems",
    system_prompt: str = "",
) -> vf.SingleTurnEnv:
    """
    Create a verifiers-compatible environment for OCaml code generation.

    The environment uses:
    - OCaml compilation and testing as the reward signal
    - Graduated rewards for partial progress (type errors, compilation)
    - Degenerate output detection to penalize prose/spam

    Args:
        dataset_id: HuggingFace dataset identifier
        system_prompt: System prompt for the model (default empty,
                      as prompts are in the dataset)

    Returns:
        SingleTurnEnv configured for OCaml code generation
    """
    # Load and transform dataset
    dataset = load_ocaml_dataset(dataset_id)

    # Create environment with OCaml reward rubric
    env = vf.SingleTurnEnv.create(
        system_prompt=system_prompt,
        rubric=[ocaml_reward],
        dataset=dataset,
    )

    return env


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data types
    "RewardResult",
    # Code extraction
    "extract_code_block",
    "extract_function_signature",
    "prepend_signature",
    "count_non_empty_code_lines",
    # Compilation functions
    "type_check_reward",
    "compile_reward",
    "tests_reward",
    # Degenerate detection
    "is_degenerate_output",
    # Reward function
    "ocaml_reward",
    # Dataset and environment
    "load_ocaml_dataset",
    "create_ocaml_env",
]
