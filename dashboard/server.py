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
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRPO Training Dashboard Server')
    parser.add_argument('--port', type=int, default=PORT, help='Port to serve on')
    parser.add_argument('--log', type=str, default=DEFAULT_LOG_FILE, help='Path to learning.log')
    args = parser.parse_args()

    LOG_FILE = args.log

    print('Starting dashboard server...')
    print(f'  Log file: {LOG_FILE}')
    print(f'  Dashboard: http://localhost:{args.port}')

    # Create dashboard directory if it doesn't exist, just in case
    if not os.path.exists(DASHBOARD_DIR):
        os.makedirs(DASHBOARD_DIR)

    with http.server.HTTPServer(('', args.port), DashboardHandler) as server:
        server.serve_forever()
