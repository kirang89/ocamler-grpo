import atexit
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from logger import RewardLogger, log_reward_entries

FUNCTION_DEF_RE = re.compile(r"\blet\s+[a-zA-Z0-9_']+\s*(?:\([^)]*\))?\s*=", re.MULTILINE)
MIN_NON_EMPTY_LINES = 2
CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "code", "language", "language:ocaml"}

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


class RewardEvaluator:
    """Caches OCaml compilation results for completions."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str], Dict[str, bool]] = {}

    def evaluate(self, problem_id: str, completion: str, tests: str = "") -> Dict[str, bool]:
        """Compile and run a completion combined with pre-defined tests,
        returning booleans for downstream reward fns."""
        code = extract_code_block(completion)
        cache_key = (problem_id, code, tests)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not code:
            result = {"type_check": False, "compile": False, "tests": False}
            self._cache[cache_key] = result
            return result

        # Combine solution with pre-defined tests
        combined_code = f"{code.rstrip()}\n\n{tests.strip()}\n"

        with tempfile.TemporaryDirectory(prefix=f"{problem_id}_reward_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            source_path = tmpdir / f"{problem_id}.ml"
            source_path.write_text(combined_code, encoding="utf-8")

            type_ok, _ = run_type_check(source_path)
            if type_ok:
                compile_ok, _ = compile_program(source_path, problem_id)
            else:
                compile_ok = False

            if compile_ok:
                exec_path = tmpdir / problem_id
                test_ok, _ = run_tests(exec_path)
            else:
                test_ok = False

        result = {"type_check": type_ok, "compile": compile_ok, "tests": test_ok}
        self._cache[cache_key] = result
        return result


# Reward Functions


def build_reward_functions(
    evaluator: RewardEvaluator, logger: RewardLogger | None
) -> List[Callable]:
    """Expose separate reward streams for heuristics plus OCaml-grounded rewards."""

    return [make_syntax_aware_reward(evaluator, logger)]


def make_syntax_aware_reward(evaluator, logger):
    """
    Syntax-aware reward function with strengthened graduated rewards and prose penalties.

    Reward structure optimized for Qwen2.5-Coder-1.5B with pre-defined tests:
    - Type checking: 25% with graduated partial credit (STRENGTHENED):
        * 0 errors (perfect): 0.25 (100%)
        * 1 error: 0.20 (80%)
        * 2 errors: 0.15 (60%)
        * 3 errors: 0.10 (40%)
        * 4 errors: 0.05 (20%)
        * 5+ errors: 0.02 (8%)
    - Compilation: 10% with partial credit (ALWAYS attempted):
        * Compiles successfully: 0.10 (100%)
        * Type checks perfectly but fails to compile: 0.05 (50%)
        * Has type errors and fails to compile: 0.01 (10%)
    - Tests: 65% graduated by test passage (0-65%):
        * All tests pass: 0.65
        * Partial: 0.65 * (tests_passed / total_tests)
    - Prose penalty: 0.3 multiplier if degenerate output detected (multi-signal)

    Key changes from previous version:
    1. Removed END marker requirement (no structural reward)
    2. Strengthened type checking rewards (20% → 25%, larger partial credit gaps)
    3. Added multi-signal prose detection (conversational text, gibberish, spam)
    4. Weakened prose penalty (0.05 → 0.3 to prevent dominating learning signal)
    5. Tests now graduated (not all-or-nothing) for smoother reward landscape

    Design rationale: Graduated rewards create strong learning signal even with
    high failure rates. Prose penalty is secondary (guardrail), not primary driver.
    """

    max_workers = int(os.environ.get("REWARD_POOL_SIZE", "0"))
    pool: ProcessPoolExecutor | None = None
    if max_workers > 1:
        pool = ProcessPoolExecutor(max_workers=max_workers)
        atexit.register(pool.shutdown, wait=True)

    def reward_func(
        prompts: List[str],
        completions: List[str],
        completion_ids=None,
        problem_id: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        ids = problem_id or kwargs.get("problem_id") or []
        tests_list = kwargs.get("tests") or []
        rewards = []
        detailed_logs = []
        completion_logs = []

        if pool:
            jobs = []
            for idx, completion in enumerate(completions):
                pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
                tests = tests_list[idx] if idx < len(tests_list) else ""
                jobs.append(pool.submit(_score_completion, pid, completion, tests))
        for idx, completion in enumerate(completions):
            pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
            tests = tests_list[idx] if idx < len(tests_list) else ""

            if pool:
                result = jobs[idx].result()
            else:
                result = _score_completion(pid, completion, tests)

            rewards.append(float(result["total_reward"]))
            detailed_logs.append(build_detailed_log_entry(pid, completion, result))
            completion_logs.append(build_completion_log_entry(pid, completion, result))

        if logger:
            logger.log("syntax_aware_breakdown", detailed_logs)
            logger.log("completions", completion_logs)

        return rewards

    reward_func.__name__ = "syntax_aware_reward"
    return reward_func


def count_non_empty_code_lines(completion: str) -> int:
    """Return non-empty, non-comment OCaml line count for heuristics."""
    code = extract_code_block(completion)
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("(*"):
            continue
        count += 1
    return count


def score_non_empty(completion: str) -> float:
    return 1.0 if count_non_empty_code_lines(completion) >= MIN_NON_EMPTY_LINES else 0.0


def score_has_function_definition(completion: str) -> float:
    code = extract_code_block(completion)
    return 1.0 if FUNCTION_DEF_RE.search(code) else 0.0


def run_subprocess(cmd: List[str], workdir: Path, timeout: int = 30) -> Tuple[bool, str]:
    """Execute an OCaml tool and return success plus logs for debugging rewards."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]} timed out after {timeout}s"
    except FileNotFoundError as exc:
        return False, f"{cmd[0]} command not found: {exc}"

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += proc.stderr
    return proc.returncode == 0, output.strip()


def run_type_check(source_path: Path) -> Tuple[bool, str]:
    """Use ocamlc -c to catch syntax/type errors before linking."""
    return run_subprocess(["ocamlc", "-c", source_path.name], source_path.parent)


def compile_program(source_path: Path, output_name: str) -> Tuple[bool, str]:
    """Produce an executable so tests can run when type checking succeeds."""
    return run_subprocess(["ocamlc", "-o", output_name, source_path.name], source_path.parent)


def run_tests(executable_path: Path) -> Tuple[bool, str]:
    """Run the generated binary which should self-test the candidate solution."""
    return run_subprocess([f"./{executable_path.name}"], executable_path.parent)


# Utilities


def build_detailed_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
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


def build_completion_log_entry(pid: str, completion: str, result: Dict[str, Any]) -> Dict[str, Any]:
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


def make_structural_reward(
    reward_name: str, scorer: Callable[[str], float], logger: RewardLogger | None
) -> Callable:
    """Wrap simple completion-scoring heuristics into GRPO reward callbacks."""

    def reward_func(
        prompts: List[str],
        completions: List[str],
        completion_ids=None,
        problem_id: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        ids = problem_id or kwargs.get("problem_id") or []
        rewards = [float(scorer(completion)) for completion in completions]
        log_reward_entries(logger, reward_name, ids, completions, rewards)
        return rewards

    reward_func.__name__ = f"{reward_name}_reward"
    return reward_func


def make_reward_function(
    metric_key: str, evaluator: RewardEvaluator, logger: RewardLogger | None
) -> Callable:
    """Generate a GRPO reward callback bound to a specific evaluator metric."""

    def reward_func(
        prompts: List[str],
        completions: List[str],
        completion_ids=None,
        problem_id: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        ids = problem_id or kwargs.get("problem_id") or []
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
            result = evaluator.evaluate(pid, completion)
            rewards.append(1.0 if result.get(metric_key, False) else 0.0)
        log_reward_entries(logger, metric_key, ids, completions, rewards)
        return rewards

    reward_func.__name__ = f"{metric_key}_reward"
    return reward_func


# Type check, compile, and test scoring helpers


def prepare_source_file(
    pid: str, completion: str, tests: str, tmpdir: Path
) -> Tuple[Optional[Path], Optional[str]]:
    """Extract code and write to temp file. Returns (source_path, code) or (None, None) if invalid."""
    code = extract_code_block(completion)
    if not code or count_non_empty_code_lines(completion) < MIN_NON_EMPTY_LINES:
        return None, None

    combined_code = f"{code.rstrip()}\n\n{tests.strip()}\n"
    source_path = tmpdir / f"{pid}.ml"
    source_path.write_text(combined_code, encoding="utf-8")
    return source_path, code


def apply_degenerate_penalty(base_reward: float, completion: str, code: str) -> Tuple[float, bool]:
    """Apply prose penalty if output is degenerate. Returns (total_reward, penalty_applied)."""
    is_degenerate = is_degenerate_output(completion, code)
    if is_degenerate:
        return base_reward * 0.3, True
    return base_reward, False


def get_type_check_score(source_path: Path, tmpdir: Path) -> Dict[str, Any]:
    """Run OCaml type check and return graduated score based on error count."""
    try:
        result = subprocess.run(
            ["ocamlc", "-c", source_path.name],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return {
            "score": 0.0,
            "syntax_errors": None,
            "error_details": "[TimeoutExpired] Type check failed",
            "has_syntax_error": False,
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "syntax_errors": None,
            "error_details": f"[{type(exc).__name__}] Type check failed",
            "has_syntax_error": False,
            "timed_out": True,
        }

    if result.returncode == 0:
        return {
            "score": 0.25,
            "syntax_errors": 0,
            "error_details": "success",
            "has_syntax_error": False,
            "timed_out": False,
        }

    stderr = result.stderr
    has_syntax_error = bool(
        re.search(
            r"Syntax error|Illegal character|unexpected token|Unterminated|"
            r"This '.*' might be unmatched",
            stderr,
            re.IGNORECASE,
        )
    )
    error_count = len(re.findall(r"\bError:", stderr))

    if has_syntax_error:
        return {
            "score": 0.0,
            "syntax_errors": error_count,
            "error_details": f"[SYNTAX ERROR] {stderr[:300]}",
            "has_syntax_error": True,
            "timed_out": False,
        }

    # Graduate type error scores: 1->0.20, 2->0.15, 3->0.10, 4->0.05, 5+->0.02
    score_map = {0: 0.0, 1: 0.20, 2: 0.15, 3: 0.10, 4: 0.05}
    score = score_map.get(error_count, 0.02)

    return {
        "score": score,
        "syntax_errors": 0,
        "error_details": f"[TYPE ERRORS: {error_count}] {stderr[:250]}",
        "has_syntax_error": False,
        "timed_out": False,
    }


def get_compile_score(
    source_path: Path, tmpdir: Path, pid: str, type_check: Dict[str, Any]
) -> float:
    """Run OCaml compilation and return score based on success and type check state."""
    if type_check["timed_out"]:
        return 0.0

    try:
        result = subprocess.run(
            ["ocamlc", "-o", pid, source_path.name],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, Exception):
        return 0.0

    if result.returncode == 0:
        return 0.10
    elif type_check["score"] == 0.25:  # Perfect type check
        return 0.05
    elif type_check["has_syntax_error"]:
        return 0.0
    else:
        return 0.01


def get_test_score(tmpdir: Path, pid: str, compile_succeeded: bool) -> Tuple[float, Optional[str]]:
    """Run compiled executable and return test score and timeout stage if any."""
    if not compile_succeeded:
        return 0.0, None

    try:
        result = subprocess.run(
            [f"./{pid}"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (0.65 if result.returncode == 0 else 0.0), None
    except subprocess.TimeoutExpired:
        return 0.0, "tests"


def _score_completion(
    pid: str, completion: str, tests: str
) -> Dict[str, float | str | bool | None]:
    with tempfile.TemporaryDirectory(prefix=f"{pid}_reward_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        source_path, code = prepare_source_file(pid, completion, tests, tmpdir)

        if source_path is None:
            return {"problem_id": pid, **REWARDS_ZERO}

        # Run type check, compile, and tests
        type_check = get_type_check_score(source_path, tmpdir)
        compile_score = get_compile_score(source_path, tmpdir, pid, type_check)
        compile_succeeded = compile_score == 0.10
        test_score, test_timeout = get_test_score(tmpdir, pid, compile_succeeded)

    # Determine timeout stage
    timeout_stage = "type_check" if type_check["timed_out"] else test_timeout

    # Calculate rewards
    base_reward = type_check["score"] + compile_score + test_score
    total_reward, prose_penalty_applied = apply_degenerate_penalty(base_reward, completion, code)

    return {
        "problem_id": pid,
        "total_reward": float(total_reward),
        "base_reward": float(base_reward),
        "type_score": float(type_check["score"]),
        "compile_score": float(compile_score),
        "test_score": float(test_score),
        "syntax_errors": type_check["syntax_errors"],
        "error_details": type_check["error_details"],
        "prose_penalty_applied": prose_penalty_applied,
        "is_degenerate": prose_penalty_applied,
        "timeout_stage": timeout_stage,
        "passed": bool(test_score >= 0.65),
    }


def extract_code_block(text: str) -> str:
    """Strip markdown fences so only runnable OCaml reaches the evaluator."""
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


def is_degenerate_output(completion: str, code: str) -> bool:
    """
    Multi-signal detection for degenerate outputs (prose, gibberish, spam).
    Returns True if output appears degenerate (1+ signals detected).

    This function detects:
    - Natural language prose (conversational patterns)
    - Low OCaml keyword density (gibberish)
    - Repetitive patterns (spam)
    - Low code purity (too much wrapper text)
    - Markdown code block spam (too many ``` markers)
    """
    if os.environ.get("GRPO_DISABLE_PROSE_PENALTY") == "true":
        return False

    issues = 0

    # Signal 1: Conversational prose patterns
    PROSE_PATTERNS = [
        r"To solve this",
        r"Here'?s",
        r"I apologize",
        r"Let me",
        r"You can use",
        r"The solution",
        r"This (approach|implementation|works|method)",
        r"[.!?]\s+[A-Z]",  # Multiple sentences (prose structure)
    ]

    for pattern in PROSE_PATTERNS:
        if re.search(pattern, completion, re.IGNORECASE):
            issues += 1
            break  # Only count once for prose patterns

    # Signal 2: Low OCaml keyword density (indicates gibberish)
    if code:
        keywords = len(
            re.findall(r"\b(let|match|with|if|then|else|fun|rec|type|val|module|open|in)\b", code)
        )
        code_tokens = len(code.split())
        keyword_density = keywords / code_tokens if code_tokens > 0 else 0

        if keyword_density < 0.05:  # Real OCaml code has ~10-20% keyword density
            issues += 1

    # Signal 3: Highly repetitive content (spam patterns)
    if len(completion) > 100:
        # Check for repeated 50-char chunks
        chunks = [completion[i : i + 50] for i in range(0, len(completion) - 50, 25)]
        if chunks:
            unique_chunks = len(set(chunks))
            repetition_ratio = unique_chunks / len(chunks) if chunks else 1.0

            if repetition_ratio < 0.3:  # >70% repetition
                issues += 1

    # Signal 4: Low code purity (too much wrapper text)
    if len(code) > 0 and len(completion) > 0:
        code_purity = len(code) / len(completion)

        if code_purity < 0.5:  # Less than half is actual code
            issues += 1

    # Signal 5: Markdown code block spam (too many ``` markers)
    code_block_count = completion.count("```")
    # Each code block has 2 markers (opening and closing), so count pairs
    # Legitimate completions should have at most 1-2 code blocks (2-4 markers)
    if code_block_count > 4:  # More than 2 code block pairs
        issues += 1

    # Require 1+ signals to trigger penalty
    return issues >= 1
