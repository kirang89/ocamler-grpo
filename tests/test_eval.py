#!/usr/bin/env python3
"""Tests for eval/eval.py"""

import pytest

from eval.constants import FAILURE_STAGE_PATTERNS
from eval.eval import _base_scores, build_result, map_reason_to_failure_stage


class TestMapReasonToFailureStage:
    """Tests for map_reason_to_failure_stage function."""

    def test_none_returns_empty(self):
        assert map_reason_to_failure_stage(None) == ""

    def test_empty_string_returns_other(self):
        result = map_reason_to_failure_stage("")
        assert result.startswith("other:")

    def test_syntax_error(self):
        assert map_reason_to_failure_stage("Syntax error in file.ml") == "type_check:syntax"
        assert map_reason_to_failure_stage("SYNTAX ERROR") == "type_check:syntax"

    def test_type_error(self):
        assert map_reason_to_failure_stage("Type error: expected int") == "type_check:type"
        assert map_reason_to_failure_stage("TYPE ERROR") == "type_check:type"

    def test_type_check_timeout(self):
        assert map_reason_to_failure_stage("timeout (type_check)") == "type_check:timeout"

    def test_compile_failure(self):
        assert map_reason_to_failure_stage("Compile failure") == "compile"

    def test_compile_timeout(self):
        assert map_reason_to_failure_stage("timeout (compile)") == "compile:timeout"

    def test_unbound_module(self):
        assert map_reason_to_failure_stage("Unbound module Str") == "compile:unbound_module"

    def test_unbound_value(self):
        assert map_reason_to_failure_stage("Unbound value foo") == "compile:unbound_value"

    def test_test_failure(self):
        assert map_reason_to_failure_stage("Test failure: expected 5 got 3") == "execution:test_fail"

    def test_test_timeout(self):
        assert map_reason_to_failure_stage("timeout (tests)") == "execution:timeout"

    def test_exception(self):
        assert map_reason_to_failure_stage("Exception: Division_by_zero") == "execution:exception"
        assert map_reason_to_failure_stage("EXCEPTION occurred") == "execution:exception"

    def test_fatal_error(self):
        assert map_reason_to_failure_stage("Fatal error: out of memory") == "execution:exception"

    def test_style_prefix(self):
        assert map_reason_to_failure_stage("style: trailing whitespace") == "style"
        assert map_reason_to_failure_stage("Style: mixed tabs") == "style"

    def test_degenerate_repetitive(self):
        assert map_reason_to_failure_stage("repetitive output detected") == "degenerate:repetitive"

    def test_degenerate_low_code_ratio(self):
        assert map_reason_to_failure_stage("low code ratio") == "degenerate:low_code_ratio"
        assert map_reason_to_failure_stage("code purity check failed") == "degenerate:low_code_ratio"

    def test_degenerate_code_block_spam(self):
        assert map_reason_to_failure_stage("code block spam") == "degenerate:code_block_spam"

    def test_degenerate_stub(self):
        assert map_reason_to_failure_stage("stub implementation") == "degenerate:stub"

    def test_degenerate_empty(self):
        assert map_reason_to_failure_stage("empty output") == "degenerate:empty"
        assert map_reason_to_failure_stage("too short") == "degenerate:empty"

    def test_unknown_reason_returns_other_with_prefix(self):
        result = map_reason_to_failure_stage("something completely unknown happened here")
        assert result.startswith("other:")
        assert "something completely unknown" in result

    def test_unknown_reason_truncated_to_30_chars(self):
        long_reason = "a" * 100
        result = map_reason_to_failure_stage(long_reason)
        # "other:" is 6 chars, plus 30 chars of reason
        assert result == "other:" + "a" * 30

    def test_case_insensitive(self):
        assert map_reason_to_failure_stage("SYNTAX ERROR") == "type_check:syntax"
        assert map_reason_to_failure_stage("Syntax Error") == "type_check:syntax"
        assert map_reason_to_failure_stage("syntax error") == "type_check:syntax"

    def test_all_patterns_in_constant(self):
        """Verify all patterns in FAILURE_STAGE_PATTERNS are handled."""
        for pattern, expected_stage in FAILURE_STAGE_PATTERNS:
            result = map_reason_to_failure_stage(f"Error: {pattern} occurred")
            assert result == expected_stage, f"Pattern '{pattern}' should map to '{expected_stage}', got '{result}'"


class TestBaseScores:
    """Tests for _base_scores helper function."""

    def test_none_returns_zeros_and_generation_error(self):
        result = _base_scores(None)

        assert result["total_reward"] == 0.0
        assert result["base_reward"] == 0.0
        assert result["type_score"] == 0.0
        assert result["compile_score"] == 0.0
        assert result["test_score"] == 0.0
        assert result["failure_stage"] == "generation_error"

    def test_with_eval_result_extracts_scores(self):
        eval_result = {
            "total_reward": 0.85,
            "base_reward": 0.75,
            "type_score": 0.25,
            "compile_score": 0.10,
            "test_score": 0.50,
            "reason": "Test failure: expected 5",
        }
        result = _base_scores(eval_result)

        assert result["total_reward"] == 0.85
        assert result["base_reward"] == 0.75
        assert result["type_score"] == 0.25
        assert result["compile_score"] == 0.10
        assert result["test_score"] == 0.50
        assert result["failure_stage"] == "execution:test_fail"

    def test_with_eval_result_no_reason(self):
        eval_result = {
            "total_reward": 1.0,
            "base_reward": 1.0,
            "type_score": 0.25,
            "compile_score": 0.10,
            "test_score": 0.65,
        }
        result = _base_scores(eval_result)

        assert result["failure_stage"] == ""


class TestBuildResult:
    """Tests for build_result function."""

    def test_error_case_with_none_eval_result(self):
        result = build_result("problem_1", "easy", None)

        assert result["id"] == "problem_1"
        assert result["difficulty"] == "easy"
        assert result["total_reward"] == 0.0
        assert result["base_reward"] == 0.0
        assert result["type_score"] == 0.0
        assert result["compile_score"] == 0.0
        assert result["test_score"] == 0.0
        assert result["failure_stage"] == "generation_error"
        assert result["generation_time_sec"] == 0.0
        assert result["completion_length"] == 0

    def test_success_case_with_eval_result(self):
        eval_result = {
            "total_reward": 1.0,
            "base_reward": 1.0,
            "type_score": 0.25,
            "compile_score": 0.10,
            "test_score": 0.65,
            "reason": None,
        }
        result = build_result("problem_2", "medium", eval_result, generation_time=2.567, completion="let x = 1")

        assert result["id"] == "problem_2"
        assert result["difficulty"] == "medium"
        assert result["total_reward"] == 1.0
        assert result["base_reward"] == 1.0
        assert result["type_score"] == 0.25
        assert result["compile_score"] == 0.10
        assert result["test_score"] == 0.65
        assert result["failure_stage"] == ""
        assert result["generation_time_sec"] == 2.57  # Rounded to 2 decimals
        assert result["completion_length"] == 9  # len("let x = 1")

    def test_failure_case_with_reason(self):
        eval_result = {
            "total_reward": 0.35,
            "base_reward": 0.35,
            "type_score": 0.25,
            "compile_score": 0.10,
            "test_score": 0.0,
            "reason": "Test failure: assertion failed",
        }
        result = build_result("problem_3", "hard", eval_result, generation_time=5.0, completion="let rec f x = f x")

        assert result["id"] == "problem_3"
        assert result["difficulty"] == "hard"
        assert result["total_reward"] == 0.35
        assert result["failure_stage"] == "execution:test_fail"
        assert result["generation_time_sec"] == 5.0
        assert result["completion_length"] == 17

    def test_generation_time_rounding(self):
        result = build_result("p", "", None, generation_time=1.999)
        assert result["generation_time_sec"] == 2.0

        result = build_result("p", "", None, generation_time=1.994)
        assert result["generation_time_sec"] == 1.99

    def test_default_values(self):
        """Test default parameter values."""
        result = build_result("p", "easy", None)

        assert result["generation_time_sec"] == 0.0
        assert result["completion_length"] == 0

    def test_result_has_all_required_fields(self):
        """Verify result contains all fields needed for CSV output."""
        required_fields = [
            "id", "difficulty", "total_reward", "base_reward",
            "type_score", "compile_score", "test_score",
            "failure_stage", "generation_time_sec", "completion_length"
        ]
        result = build_result("p", "easy", None)

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
