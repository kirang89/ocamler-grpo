"""Evaluation module for OCaml code generation."""

from .compare import compute_per_problem_comparison
from .constants import (
    COMPILE_THRESHOLD,
    PASS_THRESHOLD,
    TEST_THRESHOLD,
    TYPE_CHECK_THRESHOLD,
)
from .metrics import compute_difficulty_stats, compute_failure_stages, compute_metrics
from .report import generate_html_report

__all__ = [
    "compute_metrics",
    "compute_failure_stages",
    "compute_difficulty_stats",
    "compute_per_problem_comparison",
    "generate_html_report",
    "PASS_THRESHOLD",
    "TYPE_CHECK_THRESHOLD",
    "COMPILE_THRESHOLD",
    "TEST_THRESHOLD",
]
