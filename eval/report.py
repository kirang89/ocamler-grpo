#!/usr/bin/env python3
"""
HTML report generation for evaluation runs.

Generates a static HTML report with metrics and visualizations
using the same aesthetics as the training dashboard.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from .metrics import compute_difficulty_stats, compute_failure_stages, compute_metrics


def generate_html_report(
    results: list[dict[str, Any]],
    run_dir: Path,
    model_name: str,
    input_csv: str,
) -> None:
    """
    Generate a static HTML report for the evaluation run.

    Uses Jinja2 to render the report template with computed metrics.

    Args:
        results: List of result dictionaries
        run_dir: Directory to write the report to
        model_name: Name of the model being evaluated
        input_csv: Path to the input CSV file
    """
    if len(results) == 0:
        return

    metrics = compute_metrics(results)
    failure_stages = compute_failure_stages(results)
    difficulty_stats = compute_difficulty_stats(results)

    # Prepare chart data
    failure_stages_sorted = sorted(failure_stages.items(), key=lambda x: -x[1])
    failure_data = {
        "labels": [s[0] for s in failure_stages_sorted],
        "counts": [s[1] for s in failure_stages_sorted],
    }

    difficulty_labels = list(difficulty_stats.keys())
    difficulty_data = {
        "labels": difficulty_labels,
        "passed": [difficulty_stats[d]["passed"] for d in difficulty_labels],
        "failed": [
            difficulty_stats[d]["total"] - difficulty_stats[d]["passed"]
            for d in difficulty_labels
        ],
    }

    # Load and render template
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")

    html_content = template.render(
        model_name=model_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_csv=input_csv,
        input_file=Path(input_csv).name,
        failure_stages_sorted=failure_stages_sorted,
        failure_data=failure_data,
        difficulty_data=difficulty_data,
        **metrics,
    )

    report_path = run_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
