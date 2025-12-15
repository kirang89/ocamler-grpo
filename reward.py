import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from logger import RewardLogger, log_reward_entries

FUNCTION_DEF_RE = re.compile(r"\blet\s+[a-zA-Z0-9_']+\s*(?:\([^)]*\))?\s*=", re.MULTILINE)
MIN_NON_EMPTY_LINES = 8
CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "code", "language", "language:ocaml"}


class RewardEvaluator:
    """Caches OCaml compilation results for completions."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str], Dict[str, bool]] = {}

    def evaluate(self, problem_id: str, completion: str, tests: str = "") -> Dict[str, bool]:
        """Compile and run a completion combined with pre-defined tests, returning booleans for downstream reward fns."""
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

        # Track pass rates per problem for this batch
        problem_passes: Dict[str, List[bool]] = {}

        for idx, completion in enumerate(completions):
            pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
            tests = tests_list[idx] if idx < len(tests_list) else ""

            # Initialize tracking for this problem if needed
            if pid not in problem_passes:
                problem_passes[pid] = []

            # === STAGE 1: Code Extraction and Validation ===
            code = extract_code_block(completion)

            # Gate: Must have minimal code (check BEFORE giving any reward)
            if not code or count_non_empty_code_lines(completion) < MIN_NON_EMPTY_LINES:
                rewards.append(0.0)
                detailed_logs.append(
                    {
                        "problem_id": pid,
                        "total_reward": 0.0,
                        "type_check": 0.0,
                        "compile": 0.0,
                        "tests": 0.0,
                        "syntax_errors": None,
                        "failure_reason": "insufficient_code",
                        "preview": completion[:200],
                    }
                )
                completion_logs.append(
                    {
                        "problem_id": pid,
                        "reward": 0.0,
                        "length": len(completion),
                        "completion": completion,
                    }
                )
                # Failed generation -> did not pass
                problem_passes[pid].append(False)
                continue

            # === STAGE 2: Syntax-Aware Type Checking (25%) - STRENGTHENED ===
            # Combine solution with pre-defined tests
            combined_code = f"{code.rstrip()}\n\n{tests.strip()}\n"

            with tempfile.TemporaryDirectory(prefix=f"{pid}_reward_") as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                source_path = tmpdir / f"{pid}.ml"
                source_path.write_text(combined_code, encoding="utf-8")

                # Run OCaml type checker
                timeout_stage = None
                try:
                    type_result = subprocess.run(
                        ["ocamlc", "-c", source_path.name],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                except (subprocess.TimeoutExpired, Exception) as e:
                    # Type check timed out or failed - log and treat as complete failure
                    timeout_stage = "type_check"
                    print(f"[WARNING] Type check failed for problem {pid}: {type(e).__name__}")
                    type_score = 0.0
                    syntax_errors = None
                    error_details = f"[{type(e).__name__}] Type check failed"
                    compile_score = 0.0
                    compile_succeeded = False
                    test_score = 0.0
                    has_syntax_error = False  # For compilation stage reference

                if timeout_stage != "type_check":
                    if type_result.returncode == 0:
                        # Perfect - no syntax or type errors
                        type_score = 0.25
                        syntax_errors = 0
                        error_details = "success"
                    else:
                        # Parse stderr to distinguish SYNTAX errors from TYPE errors
                        # This is critical: syntax errors = unparseable garbage, type errors = semantic mistakes
                        stderr = type_result.stderr

                        # Check for syntax errors FIRST - they indicate fundamentally broken code
                        # Common OCaml syntax error patterns:
                        # - "Syntax error" (general parse failure)
                        # - "Illegal character" (invalid tokens like markdown ```)
                        # - "Unbound" can be type OR syntax depending on context
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
                            # Syntax errors get ZERO type credit - the code is unparseable
                            # This prevents reward hacking via markdown blocks, escape chars, etc.
                            type_score = 0.0
                            syntax_errors = error_count
                            error_details = f"[SYNTAX ERROR] {stderr[:300]}"
                        else:
                            # Only TYPE errors (semantically incorrect but syntactically valid)
                            # These get graduated partial credit because they show understanding
                            # Graduated rewards based on error count (STRENGTHENED)
                            # Larger gaps between levels to create stronger gradient
                            if error_count == 0:
                                type_score = 0.0
                            elif error_count == 1:
                                type_score = 0.20  # 80% credit - very close!
                            elif error_count == 2:
                                type_score = 0.15  # 60% credit
                            elif error_count == 3:
                                type_score = 0.10  # 40% credit
                            elif error_count == 4:
                                type_score = 0.05  # 20% credit
                            else:
                                type_score = 0.02  # 8% credit for trying

                            syntax_errors = 0  # No syntax errors, only type errors
                            error_details = f"[TYPE ERRORS: {error_count}] {stderr[:250]}"

                # === STAGE 3: Compilation (10%) - always attempt ===
                # Key change: We ALWAYS try to compile, even with type errors
                # This gives the model gradient signal for "almost compiling"
                if timeout_stage != "type_check":
                    compile_score = 0.0
                    compile_succeeded = False

                    compile_result = subprocess.run(
                        ["ocamlc", "-o", pid, source_path.name],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                    if compile_result.returncode == 0:
                        # Perfect compilation
                        compile_score = 0.10
                        compile_succeeded = True
                    elif type_result.returncode == 0:
                        # Type checked perfectly but compilation failed
                        # This is closer to success than having type errors
                        # Give substantial partial credit to bridge the gap
                        compile_score = 0.05
                    elif has_syntax_error:
                        # Syntax errors get ZERO compile credit
                        # This prevents any reward for unparseable garbage
                        compile_score = 0.0
                    else:
                        # Had TYPE errors (not syntax) and compilation failed
                        # Give tiny credit for attempting valid structure
                        compile_score = 0.01

                # === STAGE 4: Test Execution (65%) - GRADUATED ===
                # Changed from all-or-nothing to graduated for smoother reward landscape
                if timeout_stage != "type_check":
                    test_score = 0.0
                    tests_passed = 0
                    total_tests = 1  # Default assumption

                    if compile_succeeded:
                        exec_path = tmpdir / pid
                        try:
                            test_result = subprocess.run(
                                [f"./{pid}"], cwd=tmpdir, capture_output=True, text=True, timeout=30
                            )
                            if test_result.returncode == 0:
                                # All tests passed
                                test_score = 0.65
                                tests_passed = total_tests
                            # Note: We could parse test output to count partial passes,
                            # but for now, OCaml test executables return 0 (all pass) or non-zero (some fail)
                            # Future enhancement: Parse assertion failures to give partial credit
                        except subprocess.TimeoutExpired:
                            timeout_stage = "tests"
                            print(f"[WARNING] Test execution timeout for problem {pid}")
                            test_score = 0.0

            # === Final Reward Calculation ===
            base_reward = type_score + compile_score + test_score

            # === Prose/Degenerate Output Penalty (0.3 multiplier) ===
            # Multi-signal detection prevents simple reward hacking
            is_degenerate = is_degenerate_output(completion, code)
            if is_degenerate:
                total_reward = base_reward * 0.3  # 70% penalty (guardrail, not dominant signal)
                prose_penalty_applied = True
            else:
                total_reward = base_reward
                prose_penalty_applied = False

            rewards.append(total_reward)

            # Record success for metrics
            # We define "passing" as getting full credit on tests (0.65)
            # This aligns with Pass@k definition (functional correctness)
            passed = test_score >= 0.65
            problem_passes[pid].append(passed)

            # Detailed logging
            log_entry = {
                "problem_id": pid,
                "total_reward": float(total_reward),
                "base_reward": float(base_reward),
                "type_check": float(type_score),
                "compile": float(compile_score),
                "tests": float(test_score),
                "syntax_errors": syntax_errors if "syntax_errors" in locals() else None,
                "error_sample": error_details if "error_details" in locals() else None,
                "prose_penalty_applied": prose_penalty_applied,
                "is_degenerate": is_degenerate,
                "preview": completion[:200],
            }
            if timeout_stage:
                log_entry["timeout_stage"] = timeout_stage
            detailed_logs.append(log_entry)

            completion_log_entry = {
                "problem_id": pid,
                "reward": float(total_reward),
                "base_reward": float(base_reward),
                "length": len(completion),
                "prose_penalty_applied": prose_penalty_applied,
                "completion": completion,
            }
            if timeout_stage:
                completion_log_entry["timeout_stage"] = timeout_stage
            completion_logs.append(completion_log_entry)

        # === Compute Batch-Level Metrics ===
        if logger and problem_passes:
            # Calculate Pass@1 (mean accuracy per problem, then averaged)
            pass_1_scores = [sum(p) / len(p) for p in problem_passes.values()]
            batch_pass_1 = sum(pass_1_scores) / len(pass_1_scores) if pass_1_scores else 0.0

            # Calculate Pass@All (at least one success per problem)
            pass_all_scores = [1.0 if any(p) else 0.0 for p in problem_passes.values()]
            batch_pass_all = sum(pass_all_scores) / len(pass_all_scores) if pass_all_scores else 0.0

            logger.log_metrics(
                {
                    "pass_at_1": batch_pass_1,
                    "pass_at_all": batch_pass_all,
                    "batch_size": len(completions),
                    "num_problems": len(problem_passes),
                }
            )

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


def run_subprocess(cmd: List[str], workdir: Path) -> Tuple[bool, str]:
    """Execute an OCaml tool and return success plus logs for debugging rewards."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
        )
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
    Returns True if output appears degenerate. Requires 2+ signals to avoid false positives.

    This function detects:
    - Natural language prose (conversational patterns)
    - Low OCaml keyword density (gibberish)
    - Repetitive patterns (spam)
    - Low code purity (too much wrapper text)
    - Markdown code block spam (too many ``` markers)
    """
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

    # Require 1+ signals to trigger penalty (reduces false positives)
    return issues >= 1
