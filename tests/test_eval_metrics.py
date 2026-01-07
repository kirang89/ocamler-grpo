#!/usr/bin/env python3
"""Tests for eval/metrics.py"""

import pytest

from eval.constants import (
    COMPILE_THRESHOLD,
    PASS_THRESHOLD,
    TEST_THRESHOLD,
    TYPE_CHECK_THRESHOLD,
)
from eval.metrics import compute_difficulty_stats, compute_failure_stages, compute_metrics


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_empty_results_returns_zero_total(self):
        result = compute_metrics([])
        assert result["total"] == 0
        assert result["name"] == ""

    def test_empty_results_with_name(self):
        result = compute_metrics([], name="test-model")
        assert result["total"] == 0
        assert result["name"] == "test-model"

    def test_string_to_float_conversion(self):
        """CSV loading produces strings - verify they're converted."""
        results = [
            {
                "total_reward": "1.0",
                "type_score": "0.25",
                "compile_score": "0.10",
                "test_score": "0.65",
                "generation_time_sec": 2.0,
            }
        ]
        metrics = compute_metrics(results)
        assert metrics["passed"] == 1
        assert metrics["type_check_pass"] == 1
        assert metrics["compiles"] == 1
        assert metrics["tests_pass"] == 1

    def test_pass_threshold_boundary_below(self):
        """Result just below PASS_THRESHOLD should not count as passed."""
        results = [
            {
                "total_reward": PASS_THRESHOLD - 0.001,
                "type_score": 0.25,
                "compile_score": 0.10,
                "test_score": 0.65,
                "generation_time_sec": 1.0,
            }
        ]
        metrics = compute_metrics(results)
        assert metrics["passed"] == 0
        assert metrics["pass_rate"] == 0.0

    def test_pass_threshold_boundary_exact(self):
        """Result exactly at PASS_THRESHOLD should count as passed."""
        results = [
            {
                "total_reward": PASS_THRESHOLD,
                "type_score": 0.25,
                "compile_score": 0.10,
                "test_score": 0.65,
                "generation_time_sec": 1.0,
            }
        ]
        metrics = compute_metrics(results)
        assert metrics["passed"] == 1
        assert metrics["pass_rate"] == 100.0

    def test_type_check_threshold_boundary(self):
        """Test TYPE_CHECK_THRESHOLD boundary."""
        below = [{"total_reward": 0, "type_score": TYPE_CHECK_THRESHOLD - 0.01, "compile_score": 0, "test_score": 0, "generation_time_sec": 0}]
        exact = [{"total_reward": 0, "type_score": TYPE_CHECK_THRESHOLD, "compile_score": 0, "test_score": 0, "generation_time_sec": 0}]

        assert compute_metrics(below)["type_check_pass"] == 0
        assert compute_metrics(exact)["type_check_pass"] == 1

    def test_compile_threshold_boundary(self):
        """Test COMPILE_THRESHOLD boundary."""
        below = [{"total_reward": 0, "type_score": 0, "compile_score": COMPILE_THRESHOLD - 0.01, "test_score": 0, "generation_time_sec": 0}]
        exact = [{"total_reward": 0, "type_score": 0, "compile_score": COMPILE_THRESHOLD, "test_score": 0, "generation_time_sec": 0}]

        assert compute_metrics(below)["compiles"] == 0
        assert compute_metrics(exact)["compiles"] == 1

    def test_test_threshold_boundary(self):
        """Test TEST_THRESHOLD boundary."""
        below = [{"total_reward": 0, "type_score": 0, "compile_score": 0, "test_score": TEST_THRESHOLD - 0.01, "generation_time_sec": 0}]
        exact = [{"total_reward": 0, "type_score": 0, "compile_score": 0, "test_score": TEST_THRESHOLD, "generation_time_sec": 0}]

        assert compute_metrics(below)["tests_pass"] == 0
        assert compute_metrics(exact)["tests_pass"] == 1

    def test_multiple_results_counts(self):
        """Test counting with multiple results."""
        results = [
            {"total_reward": 1.0, "type_score": 0.25, "compile_score": 0.10, "test_score": 0.65, "generation_time_sec": 1.0},
            {"total_reward": 0.5, "type_score": 0.25, "compile_score": 0.10, "test_score": 0.0, "generation_time_sec": 2.0},
            {"total_reward": 0.0, "type_score": 0.0, "compile_score": 0.0, "test_score": 0.0, "generation_time_sec": 3.0},
        ]
        metrics = compute_metrics(results)

        assert metrics["total"] == 3
        assert metrics["passed"] == 1
        assert metrics["type_check_pass"] == 2
        assert metrics["compiles"] == 2
        assert metrics["tests_pass"] == 1

    def test_pass_rate_calculation(self):
        """Test pass rate percentage calculation."""
        results = [
            {"total_reward": 1.0, "type_score": 0.25, "compile_score": 0.10, "test_score": 0.65, "generation_time_sec": 1.0},
            {"total_reward": 0.5, "type_score": 0.25, "compile_score": 0.10, "test_score": 0.0, "generation_time_sec": 1.0},
            {"total_reward": 1.0, "type_score": 0.25, "compile_score": 0.10, "test_score": 0.65, "generation_time_sec": 1.0},
            {"total_reward": 0.0, "type_score": 0.0, "compile_score": 0.0, "test_score": 0.0, "generation_time_sec": 1.0},
        ]
        metrics = compute_metrics(results)

        assert metrics["pass_rate"] == 50.0  # 2/4
        assert metrics["pass_at_1"] == 50.0

    def test_avg_reward_calculation(self):
        """Test average reward calculation."""
        results = [
            {"total_reward": 1.0, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 0},
            {"total_reward": 0.5, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 0},
            {"total_reward": 0.0, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 0},
        ]
        metrics = compute_metrics(results)

        assert metrics["avg_reward"] == 0.5  # (1.0 + 0.5 + 0.0) / 3

    def test_generation_time_stats(self):
        """Test generation time statistics."""
        results = [
            {"total_reward": 0, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 2.0},
            {"total_reward": 0, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 4.0},
            {"total_reward": 0, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 0},  # Should be excluded
        ]
        metrics = compute_metrics(results)

        assert metrics["avg_gen_time"] == 3.0  # (2.0 + 4.0) / 2
        assert metrics["total_gen_time"] == 6.0

    def test_generation_time_all_zero(self):
        """Test when all generation times are zero."""
        results = [
            {"total_reward": 0, "type_score": 0, "compile_score": 0, "test_score": 0, "generation_time_sec": 0},
        ]
        metrics = compute_metrics(results)

        assert metrics["avg_gen_time"] == 0.0
        assert metrics["total_gen_time"] == 0.0


class TestComputeFailureStages:
    """Tests for compute_failure_stages function."""

    def test_empty_results(self):
        result = compute_failure_stages([])
        assert result == {}

    def test_empty_failure_stages_ignored(self):
        """Empty string failure stages should not be counted."""
        results = [
            {"failure_stage": ""},
            {"failure_stage": "compile"},
            {"failure_stage": ""},
        ]
        stages = compute_failure_stages(results)

        assert stages == {"compile": 1}

    def test_counts_multiple_same_stage(self):
        """Multiple results with same failure stage should be counted."""
        results = [
            {"failure_stage": "compile"},
            {"failure_stage": "compile"},
            {"failure_stage": "type_check:syntax"},
        ]
        stages = compute_failure_stages(results)

        assert stages["compile"] == 2
        assert stages["type_check:syntax"] == 1

    def test_various_stages(self):
        """Test counting various failure stages."""
        results = [
            {"failure_stage": "type_check:syntax"},
            {"failure_stage": "type_check:type"},
            {"failure_stage": "compile"},
            {"failure_stage": "execution:test_fail"},
            {"failure_stage": "execution:test_fail"},
            {"failure_stage": "degenerate:empty"},
            {"failure_stage": ""},  # Should be ignored
        ]
        stages = compute_failure_stages(results)

        assert stages["type_check:syntax"] == 1
        assert stages["type_check:type"] == 1
        assert stages["compile"] == 1
        assert stages["execution:test_fail"] == 2
        assert stages["degenerate:empty"] == 1
        assert len(stages) == 5

    def test_missing_failure_stage_key(self):
        """Results without failure_stage key should use empty string."""
        results = [
            {"other_field": "value"},
            {"failure_stage": "compile"},
        ]
        stages = compute_failure_stages(results)

        assert stages == {"compile": 1}


class TestComputeDifficultyStats:
    """Tests for compute_difficulty_stats function."""

    def test_empty_results(self):
        result = compute_difficulty_stats([])
        assert result == {}

    def test_single_difficulty(self):
        results = [
            {"difficulty": "easy", "total_reward": 1.0},
            {"difficulty": "easy", "total_reward": 0.5},
        ]
        stats = compute_difficulty_stats(results)

        assert stats["easy"]["total"] == 2
        assert stats["easy"]["passed"] == 1

    def test_multiple_difficulties(self):
        results = [
            {"difficulty": "easy", "total_reward": 1.0},
            {"difficulty": "easy", "total_reward": 1.0},
            {"difficulty": "medium", "total_reward": 1.0},
            {"difficulty": "medium", "total_reward": 0.5},
            {"difficulty": "hard", "total_reward": 0.0},
        ]
        stats = compute_difficulty_stats(results)

        assert stats["easy"] == {"total": 2, "passed": 2}
        assert stats["medium"] == {"total": 2, "passed": 1}
        assert stats["hard"] == {"total": 1, "passed": 0}

    def test_missing_difficulty_defaults_to_unknown(self):
        results = [
            {"total_reward": 1.0},  # No difficulty key
            {"difficulty": None, "total_reward": 0.5},  # None value
            {"difficulty": "", "total_reward": 1.0},  # Empty string
        ]
        stats = compute_difficulty_stats(results)

        assert "unknown" in stats
        assert stats["unknown"]["total"] == 3
        assert stats["unknown"]["passed"] == 2

    def test_uses_pass_threshold(self):
        """Verify PASS_THRESHOLD is used for pass determination."""
        results = [
            {"difficulty": "test", "total_reward": PASS_THRESHOLD},
            {"difficulty": "test", "total_reward": PASS_THRESHOLD - 0.001},
        ]
        stats = compute_difficulty_stats(results)

        assert stats["test"]["total"] == 2
        assert stats["test"]["passed"] == 1
