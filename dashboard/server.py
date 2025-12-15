#!/usr/bin/env python3
import http.server
import json
import re
import os
import argparse
from collections import defaultdict
from urllib.parse import urlparse

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Serve static files from the directory where this script resides (dashboard/)
DASHBOARD_DIR = SCRIPT_DIR
# Default log file is in the parent directory
DEFAULT_LOG_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "learning.log")
PORT = 8080
# Optional batch metrics file (derived from LOG_FILE if not provided)
BATCH_METRICS_FILE = None

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
        with open(envrc_path, 'r') as f:
            for line in f:
                # Match export KEY=VALUE, handling optional quotes
                m = re.match(r'export\s+([A-Z_]+)=(.*)', line.strip())
                if m:
                    key, val = m.groups()
                    # strip comments
                    val = val.split('#')[0].strip()
                    # strip quotes if present
                    val = val.strip('"\'')
                    env_vars[key] = val
    except FileNotFoundError:
        pass 

    # 2. Read train.py for defaults
    train_py_path = os.path.join(os.path.dirname(SCRIPT_DIR), "train.py")
    train_content = ""
    try:
        with open(train_py_path, 'r') as f:
            train_content = f.read()
    except FileNotFoundError:
        pass

    # Extract/Compose the params
    # Model: check env, then look for DEFAULT_MODEL_ID constant
    model_id = env_vars.get("GRPO_MODEL_ID") or env_vars.get("HF_MODEL_ID")
    if not model_id:
        model_id = find_val(r'DEFAULT_MODEL_ID\s*=\s*"([^"]+)"', train_content, "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    
    # Dataset: check env, then look for os.environ.get default in code
    dataset = env_vars.get("TRAINING_DATASET")
    if not dataset:
         dataset = find_val(r'TRAINING_DATASET\s*=\s*os\.environ\.get\("[^"]+",\s*"([^"]+)"\)', train_content, "kiranpg/ocaml-training-problems")

    params["Model"] = model_id
    params["Dataset"] = dataset
    
    # Hyperparams: check env, then look for os.environ.get defaults
    params["Generations"] = env_vars.get("GRPO_NUM_GENERATIONS", find_val(r'GRPO_NUM_GENERATIONS", "(\d+)"', train_content, "4"))
    params["Temperature"] = env_vars.get("GRPO_TEMPERATURE", find_val(r'GRPO_TEMPERATURE", "([0-9.]+)"', train_content, "0.7"))
    params["Max Prompt"] = env_vars.get("GRPO_MAX_PROMPT", find_val(r'GRPO_MAX_PROMPT", "(\d+)"', train_content, "512"))
    params["Max Completion"] = env_vars.get("GRPO_MAX_COMPLETION", find_val(r'GRPO_MAX_COMPLETION", "(\d+)"', train_content, "512"))
    params["LR"] = env_vars.get("GRPO_LEARNING_RATE", find_val(r'GRPO_LEARNING_RATE", "([^"]+)"', train_content, "5e-6"))
    params["Beta"] = env_vars.get("GRPO_BETA", find_val(r'GRPO_BETA", "([^"]+)"', train_content, "0.0"))
    
    # LoRA
    lora_r = env_vars.get("LORA_R", find_val(r'LORA_R", "(\d+)"', train_content, "32"))
    lora_alpha = env_vars.get("LORA_ALPHA", find_val(r'LORA_ALPHA", "(\d+)"', train_content, "64"))
    lora_dropout = env_vars.get("LORA_DROPOUT", find_val(r'LORA_DROPOUT", "([0-9.]+)"', train_content, "0.05"))
    
    params["LoRA"] = f"r={lora_r}, a={lora_alpha}, d={lora_dropout}"
    
    CACHED_PARAMS = params
    return params

def parse_log_file(log_path):
    """
    Parse learning.log, keeping only complete rows (with reward data).
    Aggregate all metrics per epoch using mean.
    """
    # Regex for complete rows only (must have reward field)
    pattern = re.compile(
        r'\[Epoch (\d+\.\d+)\]\s+'
        r'loss=([^\s]+)\s+'
        r'grad=([^\s]+)\s+'
        r'lr=([^\s]+)\s+'
        r'reward=([^±]+)±([^\s]+)\s+'
        r'syntax_rew=([^±]+)±([^\s]+)\s+'
        r'entropy=([^\s]+)\s+'
        r'frac_zero_std=([^\s]+)'
    )

    # Collect all values per epoch
    epoch_data = defaultdict(lambda: {
        'loss': [], 'grad': [], 'lr': [], 'entropy': [],
        'reward_mean': [], 'reward_std': [],
        'syntax_reward_mean': [], 'syntax_reward_std': [],
        'frac_zero_std': []
    })

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    epoch = float(match.group(1))
                    epoch_data[epoch]['loss'].append(float(match.group(2)))
                    epoch_data[epoch]['grad'].append(float(match.group(3)))
                    epoch_data[epoch]['lr'].append(float(match.group(4)))
                    epoch_data[epoch]['reward_mean'].append(float(match.group(5)))
                    epoch_data[epoch]['reward_std'].append(float(match.group(6)))
                    epoch_data[epoch]['syntax_reward_mean'].append(float(match.group(7)))
                    epoch_data[epoch]['syntax_reward_std'].append(float(match.group(8)))
                    epoch_data[epoch]['entropy'].append(float(match.group(9)))
                    epoch_data[epoch]['frac_zero_std'].append(float(match.group(10)))
    except FileNotFoundError:
        print(f"Warning: Log file {log_path} not found.")
        return {'epochs': [], 'error': 'Log file not found'}

    # Aggregate: compute mean per epoch
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0

    sorted_epochs = sorted(epoch_data.keys())

    result = {
        'epochs': sorted_epochs,
        'latest_epoch': sorted_epochs[-1] if sorted_epochs else None,
        'loss': [mean(epoch_data[e]['loss']) for e in sorted_epochs],
        'grad': [mean(epoch_data[e]['grad']) for e in sorted_epochs],
        'lr': [mean(epoch_data[e]['lr']) for e in sorted_epochs],
        'entropy': [mean(epoch_data[e]['entropy']) for e in sorted_epochs],
        'reward_mean': [mean(epoch_data[e]['reward_mean']) for e in sorted_epochs],
        'reward_std': [mean(epoch_data[e]['reward_std']) for e in sorted_epochs],
        'syntax_reward_mean': [mean(epoch_data[e]['syntax_reward_mean']) for e in sorted_epochs],
        'syntax_reward_std': [mean(epoch_data[e]['syntax_reward_std']) for e in sorted_epochs],
        'frac_zero_std': [mean(epoch_data[e]['frac_zero_std']) for e in sorted_epochs],
    }

    return result


def parse_batch_metrics(metrics_path):
    """
    Parse batch_metrics.jsonl which stores pass@1 and pass@All statistics per batch.
    Returns contiguous step indexes plus metric arrays so the dashboard can plot them.
    """
    result = {
        'steps': [],
        'pass_at_1': [],
        'pass_at_all': [],
    }
    if not metrics_path:
        result['error'] = 'Batch metrics path not configured'
        return result

    step_index = 1
    try:
        with open(metrics_path, 'r') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pass_at_1 = entry.get('pass_at_1')
                pass_at_all = entry.get('pass_at_all')
                if pass_at_1 is None or pass_at_all is None:
                    continue

                result['steps'].append(step_index)
                result['pass_at_1'].append(pass_at_1)
                result['pass_at_all'].append(pass_at_all)
                step_index += 1
    except FileNotFoundError:
        result['error'] = f"Batch metrics file {metrics_path} not found."
    except OSError as exc:
        result['error'] = f"Error reading batch metrics: {exc}"

    return result

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = parse_log_file(LOG_FILE)
            data['pass_metrics'] = parse_batch_metrics(BATCH_METRICS_FILE)
            data['training_params'] = get_training_params()
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRPO Training Dashboard Server')
    parser.add_argument('--port', type=int, default=PORT, help='Port to serve on')
    parser.add_argument('--log', type=str, default=DEFAULT_LOG_FILE, help='Path to learning.log')
    parser.add_argument(
        '--batch-metrics',
        type=str,
        default=None,
        help='Path to reward_logs/batch_metrics.jsonl (defaults next to log file)',
    )
    args = parser.parse_args()

    LOG_FILE = args.log
    if args.batch_metrics:
        BATCH_METRICS_FILE = args.batch_metrics
    else:
        log_dir = os.path.dirname(LOG_FILE)
        BATCH_METRICS_FILE = os.path.join(log_dir, "reward_logs", "batch_metrics.jsonl")

    print('Starting dashboard server...')
    print(f'  Log file: {LOG_FILE}')
    print(f'  Batch metrics: {BATCH_METRICS_FILE}')
    print(f'  Dashboard: http://localhost:{args.port}')

    # Create dashboard directory if it doesn't exist, just in case
    if not os.path.exists(DASHBOARD_DIR):
        os.makedirs(DASHBOARD_DIR)

    with http.server.HTTPServer(('', args.port), DashboardHandler) as server:
        server.serve_forever()
