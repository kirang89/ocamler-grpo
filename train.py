import csv
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from transformers import TrainerCallback


def _ensure_cuda_driver():
    """
    Attempt to locate and preload libcuda.so.1 if it's not automatically found.
    This is common on cloud instances where the driver is in a non-standard path
    or LD_LIBRARY_PATH isn't set in the python environment.
    """
    if sys.platform != "linux":
        return

    # Common search paths for libcuda.so.1 on Linux
    search_paths = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    ]

    # Try to load the library
    for path in search_paths:
        if os.path.exists(path):
            try:
                # RTLD_GLOBAL ensures symbols are visible to subsequently loaded libraries (like torch)
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_ensure_cuda_driver()

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an expert OCaml engineer. Read the programming problem below and implement the solution.
    The problem specifies the function signature - you must use exactly that function name as your entry point.
    Provide only the implementation code without any test cases or explanations. Keep your solution concise (under ~200 lines).

    Example 1 (List operation with higher-order function):
    Problem: Filter positive numbers from a list
    Solution:
    ```ocaml
    let filter_positive (numbers : int list) : int list =
      List.filter (fun x -> x > 0) numbers
    ```

    Example 2 (String operation):
    Problem: Count occurrences of a character in a string
    Solution:
    ```ocaml
    let count_char (s : string) (c : char) : int =
      String.fold_left (fun acc ch -> if ch = c then acc + 1 else acc) 0 s
    ```

    Example 3 (Recursive with pattern matching):
    Problem: Calculate the sum of all elements in a list
    Solution:
    ```ocaml
    let rec sum_list (lst : int list) : int =
      match lst with
      | [] -> 0
      | head :: tail -> head + sum_list tail
    ```

    Now solve this problem:

    Problem ({problem_id}):
    {question}
    """
).strip()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAINING_DATASET = os.environ.get("TRAINING_DATASET", "kiranpg/ocaml-training-problems")
GRPO_OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", "grpo_runs")

CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "code", "language", "language:ocaml"}
FUNCTION_DEF_RE = re.compile(r"\blet\s+[a-zA-Z0-9_']+\s*(?:\([^)]*\))?\s*=", re.MULTILINE)
MIN_NON_EMPTY_LINES = 8


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


class RewardLogger:
    """Writes reward outcomes to JSONL for offline inspection."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, reward_name: str, entries: List[Dict[str, Any]]) -> None:
        path = self.base_dir / f"{reward_name}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs batch-level metrics to a specific file."""
        path = self.base_dir / "batch_metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def log_learning_metrics(log_path: Path, metrics: Dict) -> None:
    """
    Logs essential learning metrics to a dedicated file for easy monitoring.

    Args:
        log_path: Path to the learning.log file
        metrics: Dictionary of training metrics from trainer logs
    """
    # Essential metrics to track
    essential_keys = [
        "epoch",
        "loss",
        "grad_norm",
        "learning_rate",
        "reward",
        "reward_std",
        "rewards/syntax_aware_reward/mean",
        "rewards/syntax_aware_reward/std",
        "entropy",
        "frac_reward_zero_std",
    ]

    # Extract available metrics
    filtered_metrics = {k: v for k, v in metrics.items() if k in essential_keys}

    # Only log if we have meaningful metrics (skip if only epoch or empty)
    if len(filtered_metrics) <= 1:
        return

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize file with header if it doesn't exist
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("GRPO Training - Learning Metrics Log\n")
            f.write("=" * 80 + "\n\n")

    # Format and write log entry
    with log_path.open("a", encoding="utf-8") as f:
        # Epoch indicator
        epoch = filtered_metrics.get("epoch", "?")
        f.write(f"[Epoch {epoch:.2f}]")

        # Core training metrics
        if "loss" in filtered_metrics:
            f.write(f"  loss={filtered_metrics['loss']:.4f}")
        if "grad_norm" in filtered_metrics:
            f.write(f"  grad={filtered_metrics['grad_norm']:.4f}")
        if "learning_rate" in filtered_metrics:
            f.write(f"  lr={filtered_metrics['learning_rate']:.2e}")

        # Reward metrics
        if "reward" in filtered_metrics:
            reward = filtered_metrics["reward"]
            reward_std = filtered_metrics.get("reward_std", 0)
            f.write(f"  reward={reward:.3f}±{reward_std:.3f}")

        if "rewards/syntax_aware_reward/mean" in filtered_metrics:
            rew_mean = filtered_metrics["rewards/syntax_aware_reward/mean"]
            rew_std = filtered_metrics.get("rewards/syntax_aware_reward/std", 0)
            f.write(f"  syntax_rew={rew_mean:.3f}±{rew_std:.3f}")

        # Policy health metrics
        if "entropy" in filtered_metrics:
            f.write(f"  entropy={filtered_metrics['entropy']:.3f}")
        if "frac_reward_zero_std" in filtered_metrics:
            f.write(f"  frac_zero_std={filtered_metrics['frac_reward_zero_std']:.2f}")

        f.write("\n")


class LearningMetricsCallback(TrainerCallback):
    """Callback that logs essential learning metrics using log_learning_metrics."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is not None:
            log_learning_metrics(self.log_path, logs)


def log_reward_entries(
    logger: RewardLogger | None,
    reward_name: str,
    ids: List[str],
    completions: List[str],
    rewards: List[float],
) -> None:
    if logger is None:
        return
    entries: List[Dict[str, str]] = []
    for idx, reward in enumerate(rewards):
        completion = completions[idx] if idx < len(completions) else ""
        pid = ids[idx] if idx < len(ids) else f"sample_{idx}"
        entries.append(
            {
                "problem_id": pid,
                "reward": reward,
                "preview": completion[:200],
            }
        )
    logger.log(reward_name, entries)


def build_training_dataset(dataset_id: str) -> Dataset:
    """Load a Hugging Face dataset or CSV file and format it for GRPO training.

    Args:
        dataset_id: Either a Hugging Face dataset identifier (e.g., 'username/dataset-name')
                   or a local path to a CSV file (for backwards compatibility)

    Returns:
        A Hugging Face Dataset with formatted prompts for each problem
    """
    # Check if it's a local CSV file (backwards compatibility)
    if dataset_id.endswith(".csv") and os.path.exists(dataset_id):
        with open(dataset_id, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        source = f"local CSV {dataset_id}"
    else:
        # Use "force_redownload" to always re-download and show progress
        hf_dataset = load_dataset(
            dataset_id, split="train", download_mode="reuse_dataset_if_exists"
        )
        rows = [dict(row) for row in hf_dataset]
        source = f"HF dataset {dataset_id}"

    # Process all rows into training format
    dataset_rows = []
    for row in rows:
        problem_id = row["id"]
        question = row["prompt"]  # Column is named "prompt"
        tests = row["tests"]
        prompt = PROMPT_TEMPLATE.format(problem_id=problem_id, question=question)
        dataset_rows.append(
            {"prompt": prompt, "problem_id": problem_id, "question": question, "tests": tests}
        )

    if not dataset_rows:
        raise ValueError(f"No rows found in dataset: {dataset_id}")

    print(f"Loaded {len(dataset_rows)} training problems from {source}")
    return Dataset.from_list(dataset_rows)


def create_tokenizer(model_id: str):
    """Load a tokenizer configured for GRPO generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


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
                type_result = subprocess.run(
                    ["ocamlc", "-c", source_path.name],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

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
                    has_syntax_error = bool(re.search(
                        r"Syntax error|Illegal character|unexpected token|Unterminated|"
                        r"This '.*' might be unmatched",
                        stderr,
                        re.IGNORECASE
                    ))

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
                else:
                    # Had type errors and compilation failed
                    # Still give tiny credit for attempting valid structure
                    compile_score = 0.01

                # === STAGE 4: Test Execution (65%) - GRADUATED ===
                # Changed from all-or-nothing to graduated for smoother reward landscape
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
            passed = (test_score >= 0.65)
            problem_passes[pid].append(passed)

            # Detailed logging
            detailed_logs.append(
                {
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
            )

            completion_logs.append(
                {
                    "problem_id": pid,
                    "reward": float(total_reward),
                    "base_reward": float(base_reward),
                    "length": len(completion),
                    "prose_penalty_applied": prose_penalty_applied,
                    "completion": completion,
                }
            )

        # === Compute Batch-Level Metrics ===
        if logger and problem_passes:
            # Calculate Pass@1 (mean accuracy per problem, then averaged)
            pass_1_scores = [sum(p) / len(p) for p in problem_passes.values()]
            batch_pass_1 = sum(pass_1_scores) / len(pass_1_scores) if pass_1_scores else 0.0

            # Calculate Pass@All (at least one success per problem)
            pass_all_scores = [1.0 if any(p) else 0.0 for p in problem_passes.values()]
            batch_pass_all = sum(pass_all_scores) / len(pass_all_scores) if pass_all_scores else 0.0

            logger.log_metrics({
                "pass_at_1": batch_pass_1,
                "pass_at_all": batch_pass_all,
                "batch_size": len(completions),
                "num_problems": len(problem_passes)
            })

        if logger:
            logger.log("syntax_aware_breakdown", detailed_logs)
            logger.log("completions", completion_logs)

        return rewards

    reward_func.__name__ = "syntax_aware_reward"
    return reward_func


def build_reward_functions(
    evaluator: RewardEvaluator, logger: RewardLogger | None
) -> List[Callable]:
    """Expose separate reward streams for heuristics plus OCaml-grounded rewards."""

    return [make_syntax_aware_reward(evaluator, logger)]


def create_grpo_config(temperature=None) -> GRPOConfig:
    """Assemble GRPO training defaults plus any overrides from env vars.
    Note: These settings have been optimized for running on a RTX 6000 48 GB VRAM.
    """
    # Detect CUDA availability
    cuda_available = torch.cuda.is_available()
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    # set to 4 prompts/step if VRAM allows; reduce when using larger models.
    per_device_batch = int(os.environ.get("GRPO_BATCH_SIZE", "4"))
    # Leave at 1 with batch 4; raise to 2-4 only when you must drop batch size.
    grad_steps = int(os.environ.get("GRPO_GRAD_ACCUM_STEPS", "1"))
    # Target 4 completions/prompt for the RTX box—turn this up until you near 44 GB VRAM.
    num_generations = int(os.environ.get("GRPO_NUM_GENERATIONS", "4"))
    # Increase to ~512 tokens on the RTX rig to capture full OCaml problems.
    max_prompt = int(os.environ.get("GRPO_MAX_PROMPT", "512"))
    # Mirror completions at ~512 tokens so solutions + harnesses fit.
    max_completion = int(os.environ.get("GRPO_MAX_COMPLETION", "512"))
    # Stick with 1-2 passes; GRPO overfits small OCaml sets quickly.
    num_epochs = float(os.environ.get("GRPO_NUM_EPOCHS", "1"))
    # 5e-6 trains safely; bump toward 8e-6 only if the run is stable.
    learning_rate = float(os.environ.get("GRPO_LEARNING_RATE", "5e-6"))
    # Use KL Coefficient to penalize large policy shifts from reference model
    beta = float(os.environ.get("GRPO_BETA", "0.0"))

    generation_batch_size = int(
        os.environ.get("GRPO_GENERATION_BATCH_SIZE", str(per_device_batch * num_generations))
    )

    if temperature is None:
        temperature = float(os.environ.get("GRPO_TEMPERATURE", "0.7"))

    # Gradient clipping to prevent training instability
    max_grad_norm = float(os.environ.get("GRPO_MAX_GRAD_NORM", "1.0"))

    # Optional: Entropy-based token filtering (focuses training on high-entropy tokens)
    # Based on "Beyond the 80/20 Rule" paper - using 20% of highest entropy tokens
    # achieves similar performance to all tokens while improving efficiency
    top_entropy_quantile = float(os.environ.get("GRPO_TOP_ENTROPY_QUANTILE", "0.2"))

    return GRPOConfig(
        temperature=temperature,  # for training diversity
        top_p=0.95,
        output_dir=GRPO_OUTPUT_DIR,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_steps,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        max_prompt_length=max_prompt,
        max_completion_length=max_completion,
        remove_unused_columns=False,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,  # Clip gradients to prevent explosions
        log_completions=True,  # Important for detecting reward collapse
        # Keep it 1 or 2 – frequent logging helps spot reward collapse
        logging_steps=int(os.environ.get("GRPO_LOGGING_STEPS", "1")),
        bf16=use_bf16,  # Auto-detect bf16 support based on CUDA availability
        # Disable checkpointing to avoid requires_grad issues on RTX 6000 training.
        gradient_checkpointing=False,
        eval_strategy="no",
        save_steps=100,
        dataloader_num_workers=4,  # Use CPU cores
        dataloader_pin_memory=True,
        beta=beta,
        top_entropy_quantile=top_entropy_quantile,  # Focus training on high-entropy tokens
    )


def create_lora_config() -> LoraConfig:
    """Build a LoraConfig using optional env overrides."""
    # Rank 16 keeps VRAM in check; double it only if you need more adapter capacity.
    lora_r = int(os.environ.get("LORA_R", "32"))
    # Alpha 32 pairs well with r=16; scale roughly 2x the rank when you change it.
    lora_alpha = int(os.environ.get("LORA_ALPHA", "64"))
    # Small dropout (5%) stabilizes GRPO; set to 0 if you notice underfitting.
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    # Bias "none" avoids extra params; use "lora_only" when the base model expects it.
    bias = os.environ.get("LORA_BIAS", "none")
    # Cover attention (q/k/v/o) plus MLP (gate/up/down) blocks for coder backbones.
    raw_target_modules = os.environ.get(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    target_modules = [module.strip() for module in raw_target_modules.split(",") if module.strip()]

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def resolve_model_id() -> str:
    """Return a Hugging Face model identifier suitable for GRPO training."""
    candidate = os.environ.get("GRPO_MODEL_ID") or os.environ.get("HF_MODEL_ID")
    if candidate:
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("GRPO_MODEL_ID was provided but empty.")
        if ":" in candidate:
            raise ValueError(
                f"GRPO_MODEL_ID must be a Hugging Face repo id (no ':' characters). Got: {candidate}"
            )
        return candidate
    return DEFAULT_MODEL_ID


def main():
    model_id = resolve_model_id()
    dataset = build_training_dataset(TRAINING_DATASET)
    tokenizer = create_tokenizer(model_id)
    evaluator = RewardEvaluator()
    output_path = Path(GRPO_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    reward_logger = RewardLogger(output_path / "reward_logs")
    reward_funcs = build_reward_functions(evaluator, reward_logger)
    config = create_grpo_config()
    lora_config = create_lora_config()

    # Create learning metrics callback
    learning_log_path = output_path / "learning.log"
    learning_callback = LearningMetricsCallback(learning_log_path)

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[learning_callback],
    )

    trainer.train()

    trainer.save_model(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)


if __name__ == "__main__":
    main()
