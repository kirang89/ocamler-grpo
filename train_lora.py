"""GRPO + LoRA trainer for OCaml problem solving with OCaml-grounded rewards."""

import csv
import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an expert OCaml engineer. Read the programming problem below and craft an OCaml
    solution plus a lightweight test harness that executes when run. Output a single OCaml
    file, keep it under ~200 lines, and end the response with the exact marker `(* END *)`.
    Do not emit any prose, explanations, or trailing text after the marker.

    Problem ({problem_id}):
    {question}
    """
).strip()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAINING_PROBLEMS_FILE = os.environ.get("TRAINING_PROBLEMS_FILE", "problems.csv")
GRPO_OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", "grpo_runs")

CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "code", "language", "language:ocaml"}


class RewardEvaluator:
    """Caches OCaml compilation results for completions."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Dict[str, bool]] = {}

    def evaluate(self, problem_id: str, completion: str) -> Dict[str, bool]:
        """Compile and run a completion, returning booleans for downstream reward fns."""
        code = extract_code_block(completion)
        cache_key = (problem_id, code)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not code:
            result = {"type_check": False, "compile": False, "tests": False}
            self._cache[cache_key] = result
            return result

        with tempfile.TemporaryDirectory(prefix=f"{problem_id}_reward_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            source_path = tmpdir / f"{problem_id}.ml"
            source_path.write_text(f"{code.rstrip()}\n", encoding="utf-8")

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


def read_problems(csv_path: str) -> List[Dict[str, str]]:
    """Return rows from the curated OCaml problem CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_training_dataset(csv_path: str) -> Dataset:
    """Wrap the CSV in a Hugging Face Dataset containing prompts per problem."""
    rows = read_problems(csv_path)
    dataset_rows = []
    for row in rows:
        problem_id = row["id"]
        question = row["question"]
        prompt = PROMPT_TEMPLATE.format(problem_id=problem_id, question=question)
        dataset_rows.append({"prompt": prompt, "problem_id": problem_id, "question": question})
    if not dataset_rows:
        raise ValueError(f"No rows found in {csv_path}")
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


def make_reward_function(metric_key: str, evaluator: RewardEvaluator) -> Callable:
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
        return rewards

    reward_func.__name__ = f"{metric_key}_reward"
    return reward_func


def build_reward_functions(evaluator: RewardEvaluator) -> List[Callable]:
    """Expose separate reward streams for type check, compile, and end-to-end tests."""
    return [
        make_reward_function("type_check", evaluator),
        make_reward_function("compile", evaluator),
        make_reward_function("tests", evaluator),
    ]


def create_grpo_config() -> GRPOConfig:
    """Assemble GRPO training defaults plus any overrides from env vars.
    Note: These settings have been optimized for running on a RTX 6000 48 GB VRAM.
    """
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
    generation_batch_size = int(
        os.environ.get("GRPO_GENERATION_BATCH_SIZE", str(per_device_batch * num_generations))
    )

    return GRPOConfig(
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
        # Keep it 1 or 2 – frequent logging helps spot reward collapse; the overhead is tiny.
        logging_steps=int(os.environ.get("GRPO_LOGGING_STEPS", "1")),
        # Disable checkpointing to avoid requires_grad issues on RTX 6000 training.
        gradient_checkpointing=False,
    )


def create_lora_config() -> LoraConfig:
    """Build a LoraConfig using optional env overrides."""
    # Rank 16 keeps VRAM in check; double it only if you need more adapter capacity.
    lora_r = int(os.environ.get("LORA_R", "16"))
    # Alpha 32 pairs well with r=16; scale roughly 2x the rank when you change it.
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
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
    """Tie tokenizer/model selection, dataset prep, rewards, and LoRA together."""
    model_id = resolve_model_id()
    dataset = build_training_dataset(TRAINING_PROBLEMS_FILE)
    tokenizer = create_tokenizer(model_id)
    evaluator = RewardEvaluator()
    reward_funcs = build_reward_functions(evaluator)
    config = create_grpo_config()
    lora_config = create_lora_config()

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)


if __name__ == "__main__":
    main()
