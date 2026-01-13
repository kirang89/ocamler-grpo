#!/usr/bin/env python3
import argparse
import http.server
import json
import os
import re
from collections import defaultdict
from urllib.parse import urlparse

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Serve static files from the directory where this script resides (dashboard/)
DASHBOARD_DIR = SCRIPT_DIR
# Default metrics file is in the parent directory (GRPO uses metrics.jsonl like SFT)
DEFAULT_METRICS_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "grpo_runs/metrics.jsonl")
DEFAULT_ERROR_LOG = os.path.join(
    os.path.dirname(SCRIPT_DIR), "grpo_runs/reward_logs/syntax_aware_breakdown.jsonl"
)
DEFAULT_COMPLETIONS_FILE = os.path.join(
    os.path.dirname(SCRIPT_DIR), "grpo_runs", "reward_logs", "completions.jsonl"
)
# SFT metrics paths
DEFAULT_SFT_METRICS_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "sft_runs/metrics.jsonl")
SFT_METRICS_FILE = DEFAULT_SFT_METRICS_FILE

PORT = 8080
# GRPO metrics file
GRPO_METRICS_FILE = DEFAULT_METRICS_FILE
# Optional batch metrics file (derived from GRPO_METRICS_FILE if not provided)
BATCH_METRICS_FILE = None
ERROR_LOG_FILE = DEFAULT_ERROR_LOG
COMPLETIONS_FILE = DEFAULT_COMPLETIONS_FILE

# Cache for training parameters
CACHED_PARAMS = None


def get_training_params():
    global CACHED_PARAMS
    if CACHED_PARAMS is not None:
        return CACHED_PARAMS

    params = {}

    # Helper to find value in text via regex
    def find_val(pattern, text, default="?"):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    # 1. Read .envrc
    envrc_path = os.path.join(os.path.dirname(SCRIPT_DIR), ".envrc")
    env_vars = {}
    try:
        with open(envrc_path, "r") as f:
            for line in f:
                # Match export KEY=VALUE, handling optional quotes
                m = re.match(r"export\s+([A-Z_]+)=(.*)", line.strip())
                if m:
                    key, val = m.groups()
                    # strip comments
                    val = val.split("#")[0].strip()
                    # strip quotes if present
                    val = val.strip("\"'")
                    env_vars[key] = val
    except FileNotFoundError:
        pass

    # 2. Read train.py for defaults
    train_py_path = os.path.join(os.path.dirname(SCRIPT_DIR), "rlvr", "train.py")
    train_content = ""
    try:
        with open(train_py_path, "r") as f:
            train_content = f.read()
    except FileNotFoundError:
        pass

    # Extract/Compose the params
    # Model: check env, then look for DEFAULT_MODEL_ID constant
    model_id = env_vars.get("GRPO_MODEL_ID") or env_vars.get("HF_MODEL_ID")
    if not model_id:
        model_id = find_val(
            r'DEFAULT_MODEL_ID\s*=\s*"([^"]+)"', train_content, "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        )

    # Dataset: check env, then look for os.environ.get default in code
    dataset = env_vars.get("TRAINING_DATASET")
    if not dataset:
        dataset = find_val(
            r'TRAINING_DATASET\s*=\s*os\.environ\.get\("[^"]+",\s*"([^"]+)"\)',
            train_content,
            "kiranpg/ocaml-training-problems",
        )

    params["Model"] = model_id
    params["Dataset"] = dataset

    # Hyperparams: check env, then look for os.environ.get defaults
    params["Generations"] = env_vars.get(
        "GRPO_NUM_GENERATIONS", find_val(r'GRPO_NUM_GENERATIONS", "(\d+)"', train_content, "4")
    )
    params["Temperature"] = env_vars.get(
        "GRPO_TEMPERATURE", find_val(r'GRPO_TEMPERATURE", "([0-9.]+)"', train_content, "0.7")
    )
    params["Max Prompt"] = env_vars.get(
        "GRPO_MAX_PROMPT", find_val(r'GRPO_MAX_PROMPT", "(\d+)"', train_content, "512")
    )
    params["Max Completion"] = env_vars.get(
        "GRPO_MAX_COMPLETION", find_val(r'GRPO_MAX_COMPLETION", "(\d+)"', train_content, "512")
    )
    params["LR"] = env_vars.get(
        "GRPO_LEARNING_RATE", find_val(r'GRPO_LEARNING_RATE", "([^"]+)"', train_content, "5e-6")
    )
    params["Beta"] = env_vars.get(
        "GRPO_BETA", find_val(r'GRPO_BETA", "([^"]+)"', train_content, "0.0")
    )

    # LoRA
    lora_r = env_vars.get("LORA_R", find_val(r'LORA_R", "(\d+)"', train_content, "32"))
    lora_alpha = env_vars.get("LORA_ALPHA", find_val(r'LORA_ALPHA", "(\d+)"', train_content, "64"))
    lora_dropout = env_vars.get(
        "LORA_DROPOUT", find_val(r'LORA_DROPOUT", "([0-9.]+)"', train_content, "0.05")
    )

    params["LoRA"] = f"r={lora_r}, a={lora_alpha}, d={lora_dropout}"

    CACHED_PARAMS = params
    return params


# JSONL keys that are always present (required)
_REQUIRED_KEYS = ["loss", "grad_norm", "learning_rate", "reward_mean", "reward_std",
                  "syntax_reward_mean", "syntax_reward_std", "entropy", "frac_zero_std"]

# JSONL keys that may be absent (optional)
_OPTIONAL_KEYS = ["step_time", "mean_length", "kl"]

# Rename these JSONL keys to shorter names for the frontend API
_OUTPUT_RENAMES = {"grad_norm": "grad", "learning_rate": "lr"}


def parse_grpo_metrics(metrics_path):
    """
    Parse GRPO metrics.jsonl file.
    Aggregate all metrics per epoch using mean.
    """
    # Internal keys match JSONL keys for clarity
    all_keys = _REQUIRED_KEYS + _OPTIONAL_KEYS
    epoch_data = defaultdict(lambda: {k: [] for k in all_keys})

    try:
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                epoch = entry.get("epoch")
                if epoch is None:
                    continue

                # Required fields - use 0 as default
                for key in _REQUIRED_KEYS:
                    epoch_data[epoch][key].append(entry.get(key, 0))

                # Optional fields - only append if present
                for key in _OPTIONAL_KEYS:
                    if entry.get(key) is not None:
                        epoch_data[epoch][key].append(entry[key])

    except FileNotFoundError:
        print(f"Warning: Metrics file {metrics_path} not found.")
        return {"epochs": [], "error": "Metrics file not found"}

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0

    sorted_epochs = sorted(epoch_data.keys())

    # Build result, renaming keys for frontend where needed
    result = {
        "epochs": sorted_epochs,
        "latest_epoch": sorted_epochs[-1] if sorted_epochs else None,
    }

    for key in all_keys:
        output_key = _OUTPUT_RENAMES.get(key, key)
        result[output_key] = [mean(epoch_data[e][key]) for e in sorted_epochs]

    return result


def parse_batch_metrics(metrics_path):
    """
    Parse batch_metrics.jsonl which stores pass@1 and pass@All statistics per batch.
    Returns contiguous step indexes plus metric arrays so the dashboard can plot them.
    """
    result = {
        "steps": [],
        "pass_at_1": [],
        "pass_at_all": [],
    }
    if not metrics_path:
        result["error"] = "Batch metrics path not configured"
        return result

    step_index = 1
    try:
        with open(metrics_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pass_at_1 = entry.get("pass_at_1")
                pass_at_all = entry.get("pass_at_all")
                if pass_at_1 is None or pass_at_all is None:
                    continue

                result["steps"].append(step_index)
                result["pass_at_1"].append(pass_at_1)
                result["pass_at_all"].append(pass_at_all)
                step_index += 1
    except FileNotFoundError:
        result["error"] = f"Batch metrics file {metrics_path} not found."
    except OSError as exc:
        result["error"] = f"Error reading batch metrics: {exc}"

    return result


def resolve_error_log_path(configured_path: str) -> str | None:
    """Return the first available path for the detailed reward log."""
    candidates = []
    if configured_path:
        candidates.append(configured_path)

    log_dir = os.path.dirname(GRPO_METRICS_FILE)
    reward_log_dir = os.path.join(log_dir, "reward_logs")
    candidates.append(os.path.join(reward_log_dir, "syntax_aware_breakdown.jsonl"))

    repo_root = os.path.dirname(SCRIPT_DIR)
    candidates.append(os.path.join(repo_root, "syntax_aware_breakdown.jsonl"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def parse_error_log(log_path: str):
    """
    Parse syntax_aware_breakdown.jsonl and compute cumulative error rates.

    Returns cumulative percentages so trends are easy to spot even when noisy.
    """
    result = {
        "steps": [],
        "syntax_error_rate": [],
        "type_error_rate": [],
        "compile_error_rate": [],
        "execution_failure_rate": [],
        "total_entries": 0,
    }

    resolved = resolve_error_log_path(log_path)
    if not resolved:
        result["error"] = "Error log not found"
        return result

    def is_number(value):
        return isinstance(value, (int, float))

    def almost_equal(value: float, target: float, tol: float = 1e-6) -> bool:
        return abs(value - target) <= tol

    syntax_count = type_count = compile_count = exec_fail_count = 0
    # Track test pass rate buckets for completions that compiled
    test_buckets = {
        "0%": 0,
        "1-25%": 0,
        "26-50%": 0,
        "51-75%": 0,
        "76-99%": 0,
        "100%": 0,
    }
    entries = []

    try:
        with open(resolved, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                syntax_errors = entry.get("syntax_errors")
                type_score = entry.get("type_check")
                compile_score = entry.get("compile")
                test_score = entry.get("tests")

                syntax_error = is_number(syntax_errors) and syntax_errors != 0
                type_error = not syntax_error and is_number(type_score) and 0 < type_score < 0.25
                compile_error = (
                    not syntax_error
                    and is_number(type_score)
                    and almost_equal(type_score, 0.25)
                    and is_number(compile_score)
                    and compile_score < 0.10
                )
                execution_failure = (
                    is_number(compile_score)
                    and almost_equal(compile_score, 0.10)
                    and is_number(test_score)
                    and test_score < 0.65
                )

                # Track test pass rates for completions that compiled (compile_score == 0.10)
                if is_number(compile_score) and almost_equal(compile_score, 0.10) and is_number(test_score):
                    # test_score max is 0.65, so pass_rate = test_score / 0.65
                    pass_rate = test_score / 0.65 if test_score > 0 else 0
                    if almost_equal(pass_rate, 1.0, tol=0.001):
                        test_buckets["100%"] += 1
                    elif pass_rate > 0.75:
                        test_buckets["76-99%"] += 1
                    elif pass_rate > 0.50:
                        test_buckets["51-75%"] += 1
                    elif pass_rate > 0.25:
                        test_buckets["26-50%"] += 1
                    elif pass_rate > 0:
                        test_buckets["1-25%"] += 1
                    else:
                        test_buckets["0%"] += 1

                entries.append((syntax_error, type_error, compile_error, execution_failure))
    except FileNotFoundError:
        result["error"] = f"Error log {resolved} not found"
        return result

    total_entries = len(entries)
    if total_entries == 0:
        result["error"] = "No error entries found"
        return result

    max_points = 400
    stride = max(1, total_entries // max_points)

    for idx, (syntax_error, type_error, compile_error, execution_failure) in enumerate(
        entries, start=1
    ):
        if syntax_error:
            syntax_count += 1
        if type_error:
            type_count += 1
        if compile_error:
            compile_count += 1
        if execution_failure:
            exec_fail_count += 1

        should_record = idx % stride == 0 or idx == total_entries
        if not should_record:
            continue

        result["steps"].append(idx)
        result["syntax_error_rate"].append(round(syntax_count / idx * 100, 3))
        result["type_error_rate"].append(round(type_count / idx * 100, 3))
        result["compile_error_rate"].append(round(compile_count / idx * 100, 3))
        result["execution_failure_rate"].append(round(exec_fail_count / idx * 100, 3))

    result["total_entries"] = total_entries
    # Add final counts for pie chart
    success_count = total_entries - syntax_count - type_count - compile_count - exec_fail_count
    result["counts"] = {
        "syntax_errors": syntax_count,
        "type_errors": type_count,
        "compile_errors": compile_count,
        "execution_failures": exec_fail_count,
        "success": max(0, success_count),
    }
    # Add test pass rate distribution (for completions that compiled)
    result["test_pass_distribution"] = test_buckets
    return result


def parse_completions_jsonl(jsonl_path: str):
    entries = []
    try:
        with open(jsonl_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return {"error": f"Completions file {jsonl_path} not found."}
    except OSError as exc:
        return {"error": f"Error reading completions file: {exc}"}
    return entries


# =============================================================================
# SFT Metrics Parsing
# =============================================================================


def get_sft_training_params():
    """Extract SFT training parameters from .envrc file."""
    params = {}
    envrc_path = os.path.join(os.path.dirname(SCRIPT_DIR), ".envrc")

    try:
        with open(envrc_path, "r") as f:
            for line in f:
                m = re.match(r"export\s+([A-Z_]+)=(.*)", line.strip())
                if m:
                    key, val = m.groups()
                    val = val.split("#")[0].strip().strip("\"'")
                    if key.startswith("SFT_") or key.startswith("LORA_") or key == "BASE_MODEL_ID":
                        # Convert key to display format
                        display_key = key.replace("SFT_", "").replace("_", " ").title()
                        if key == "BASE_MODEL_ID":
                            display_key = "Model"
                        elif key.startswith("LORA_"):
                            display_key = key.replace("_", " ").title()
                        params[display_key] = val
    except FileNotFoundError:
        pass

    return params


def parse_sft_metrics(metrics_path: str):
    """
    Parse SFT metrics.jsonl file and return structured data for visualization.

    The JSONL file contains:
    - train_start event with config
    - Per-step metrics: step, epoch, loss, learning_rate, grad_norm, eval_loss (optional)
    - train_end event with summary

    Returns dict with arrays for plotting.
    """
    result = {
        "steps": [],
        "epochs": [],
        "loss": [],
        "learning_rate": [],
        "grad_norm": [],
        "timestamps": [],
        "eval_steps": [],
        "eval_loss": [],
        "train_config": None,
        "train_summary": None,
        "latest_step": None,
        "latest_epoch": None,
        "latest_loss": None,
        "latest_eval_loss": None,
        "total_entries": 0,
    }

    if not os.path.exists(metrics_path):
        result["error"] = f"SFT metrics file not found: {metrics_path}"
        return result

    try:
        with open(metrics_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Handle events
                event = entry.get("event")
                if event == "train_start":
                    result["train_config"] = {
                        "total_steps": entry.get("total_steps"),
                        "num_epochs": entry.get("num_epochs"),
                        "batch_size": entry.get("batch_size"),
                        "grad_accum_steps": entry.get("grad_accum_steps"),
                        "learning_rate": entry.get("learning_rate"),
                    }
                    continue
                elif event == "train_end":
                    result["train_summary"] = {
                        "total_steps": entry.get("total_steps"),
                        "elapsed_seconds": entry.get("elapsed_seconds"),
                        "samples_per_second": entry.get("samples_per_second"),
                    }
                    continue

                # Handle metrics entries
                step = entry.get("step")
                if step is None:
                    continue

                # Check if this is a train metrics entry (has non-null loss)
                # or an eval-only entry (loss is null, has eval_loss)
                # Skip summary entries (they have train_runtime or train_loss keys)
                loss = entry.get("loss")
                eval_loss = entry.get("eval_loss")
                is_summary = "train_runtime" in entry or "train_loss" in entry

                # Only add to main arrays if this is a training step (non-null loss, not summary)
                if loss is not None and not is_summary:
                    result["steps"].append(step)
                    result["epochs"].append(entry.get("epoch", 0))
                    result["loss"].append(loss)
                    result["learning_rate"].append(entry.get("learning_rate"))
                    result["grad_norm"].append(entry.get("grad_norm"))
                    result["timestamps"].append(entry.get("timestamp"))
                    result["total_entries"] += 1

                # Track eval_loss separately (logged in separate entries)
                if eval_loss is not None:
                    result["eval_steps"].append(step)
                    result["eval_loss"].append(eval_loss)

    except OSError as exc:
        result["error"] = f"Error reading SFT metrics: {exc}"
        return result

    # Set latest values
    if result["steps"]:
        result["latest_step"] = result["steps"][-1]
        result["latest_epoch"] = result["epochs"][-1]
        result["latest_loss"] = result["loss"][-1]

    if result["eval_loss"]:
        result["latest_eval_loss"] = result["eval_loss"][-1]

    return result


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = parse_grpo_metrics(GRPO_METRICS_FILE)
            data["pass_metrics"] = parse_batch_metrics(BATCH_METRICS_FILE)
            data["training_params"] = get_training_params()
            data["error_metrics"] = parse_error_log(ERROR_LOG_FILE)
            self.wfile.write(json.dumps(data).encode())
        elif parsed.path == "/api/completions":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = parse_completions_jsonl(COMPLETIONS_FILE)
            self.wfile.write(json.dumps(data).encode())
        elif parsed.path == "/api/sft/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = parse_sft_metrics(SFT_METRICS_FILE)
            data["training_params"] = get_sft_training_params()
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training Dashboard Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to serve on")
    parser.add_argument(
        "--metrics", type=str, default=DEFAULT_METRICS_FILE, help="Path to GRPO metrics.jsonl"
    )
    parser.add_argument(
        "--batch-metrics",
        type=str,
        default=None,
        help="Path to reward_logs/batch_metrics.jsonl (defaults next to log file)",
    )
    parser.add_argument(
        "--error-log",
        type=str,
        default=DEFAULT_ERROR_LOG,
        help="Path to syntax_aware_breakdown.jsonl (defaults to repo root or reward_logs)",
    )
    parser.add_argument(
        "--completions",
        type=str,
        default=DEFAULT_COMPLETIONS_FILE,
        help="Path to completions.jsonl (defaults to latest-run/completions.jsonl)",
    )
    parser.add_argument(
        "--sft-metrics",
        type=str,
        default=DEFAULT_SFT_METRICS_FILE,
        help="Path to SFT metrics.jsonl (defaults to sft_runs/metrics.jsonl)",
    )
    args = parser.parse_args()

    GRPO_METRICS_FILE = args.metrics
    if args.batch_metrics:
        BATCH_METRICS_FILE = args.batch_metrics
    else:
        log_dir = os.path.dirname(GRPO_METRICS_FILE)
        BATCH_METRICS_FILE = os.path.join(log_dir, "reward_logs", "batch_metrics.jsonl")
    ERROR_LOG_FILE = args.error_log
    COMPLETIONS_FILE = args.completions
    SFT_METRICS_FILE = args.sft_metrics

    print("Starting dashboard server...")
    print(f"  GRPO metrics: {GRPO_METRICS_FILE}")
    print(f"  Batch metrics: {BATCH_METRICS_FILE}")
    print(f"  Error log: {resolve_error_log_path(ERROR_LOG_FILE) or ERROR_LOG_FILE}")
    print(f"  Completions: {COMPLETIONS_FILE}")
    print(f"  SFT metrics: {SFT_METRICS_FILE}")
    print(f"  Dashboard: http://localhost:{args.port}")

    # Create dashboard directory if it doesn't exist, just in case
    if not os.path.exists(DASHBOARD_DIR):
        os.makedirs(DASHBOARD_DIR)

    with http.server.HTTPServer(("", args.port), DashboardHandler) as server:
        server.serve_forever()
