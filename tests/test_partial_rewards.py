"""Tests for partial test rewards functionality.

Tests cover:
- transform_tests_for_partial_credit: OCaml assert transformation
- parse_test_results: Parsing GRPO_TEST_RESULT output
- ASSERT_PATTERN: Balanced parentheses regex matching
- Graduated test scoring: Partial pass rate rewards
- End-to-end partial credit: Full pipeline verification
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from rlvr.environment import (
    ASSERT_PATTERN,
    transform_tests_for_partial_credit,
)
from rlvr.reward import (
    TESTS_PASS_SCORE,
    parse_test_results,
    tests_reward,
)

# ============================================================================
# transform_tests_for_partial_credit Tests
# ============================================================================


class TestTransformTestsBasic:
    """Basic transformation tests for assert → __test."""

    def test_single_assertion_transformed(self):
        """Single assert statement is transformed to __test."""
        tests = "assert (foo 1 = 2);;"
        result = transform_tests_for_partial_credit(tests)

        assert "__test (foo 1 = 2);" in result
        assert "assert" not in result.split("let __test")[1]  # No assert after header

    def test_multiple_assertions_transformed(self):
        """Multiple assert statements are all transformed."""
        tests = """let () =
        assert (foo 1 = 2);
        assert (foo 2 = 4);
        assert (foo 3 = 6);;"""
        result = transform_tests_for_partial_credit(tests)

        assert result.count("__test (") == 3
        assert "assert" not in result.split("let __test")[1]

    def test_header_inserted(self):
        """Header with __passed, __total, and __test is inserted."""
        tests = "assert (x = 1);;"
        result = transform_tests_for_partial_credit(tests)

        assert "__passed = ref 0" in result
        assert "__total = ref 0" in result
        assert "let __test cond = incr __total; if cond then incr __passed" in result

    def test_printf_inserted_before_terminator(self):
        """Printf with GRPO_TEST_RESULT is inserted before ;;."""
        tests = "assert (x = 1);;"
        result = transform_tests_for_partial_credit(tests)

        assert 'Printf.printf "GRPO_TEST_RESULT:%d/%d\\n" !__passed !__total' in result
        assert result.rstrip().endswith(";;")

    def test_no_assertions_returns_unchanged(self):
        """Test code without assertions is returned unchanged."""
        tests = 'let () = print_endline "hello";;'
        result = transform_tests_for_partial_credit(tests)

        assert result == tests
        assert "__test" not in result

    def test_empty_string_returns_empty(self):
        """Empty string returns empty string."""
        result = transform_tests_for_partial_credit("")
        assert result == ""

    def test_preserves_code_between_assertions(self):
        """Code between assertions is preserved."""
        tests = """let () =
        let x = 1 in
        assert (x = 1);
        let y = 2 in
        assert (y = 2);;"""
        result = transform_tests_for_partial_credit(tests)

        assert "let x = 1 in" in result
        assert "let y = 2 in" in result


class TestTransformTestsNestedParens:
    """Tests for handling nested parentheses in assertions."""

    def test_nested_function_calls(self):
        """Assertions with nested function calls are handled."""
        tests = "assert (foo (bar (baz x)) = 42);;"
        result = transform_tests_for_partial_credit(tests)

        assert "__test (foo (bar (baz x)) = 42);" in result

    def test_list_literals_with_semicolons(self):
        """Assertions with list literals containing semicolons are handled."""
        tests = "assert (is_reflected [[0; 0]; [1; 0]] = true);;"
        result = transform_tests_for_partial_credit(tests)

        assert "__test (is_reflected [[0; 0]; [1; 0]] = true);" in result

    def test_tuple_expressions(self):
        """Assertions with tuple expressions are handled."""
        tests = "assert (fst (1, 2) = 1);;"
        result = transform_tests_for_partial_credit(tests)

        assert "__test (fst (1, 2) = 1);" in result

    def test_deeply_nested_parens(self):
        """Deeply nested parentheses are handled correctly."""
        tests = "let x = 1 in assert ((((x = 1))));;"
        result = transform_tests_for_partial_credit(tests)

        assert "__test ((((x = 1))));" in result

    def test_mixed_brackets_and_parens(self):
        """Mixed brackets and parentheses are handled."""
        tests = "assert (List.hd [1; 2; 3] = 1);;"
        result = transform_tests_for_partial_credit(tests)

        assert "__test (List.hd [1; 2; 3] = 1);" in result


class TestTransformTestsEdgeCases:
    """Edge cases for test transformation."""

    def test_assertion_without_terminator(self):
        """Assertion without ;; still works (no Printf inserted)."""
        tests = "assert (x = 1);"
        result = transform_tests_for_partial_credit(tests)

        # Should transform but not add Printf (no ;;)
        assert "__test (x = 1);" in result
        assert "GRPO_TEST_RESULT" not in result

    def test_let_binding_wrapper(self):
        """Assertions inside let () = ... block are handled."""
        tests = """let () =
        assert (factorial 5 = 120);
        assert (factorial 0 = 1);;"""
        result = transform_tests_for_partial_credit(tests)

        assert "__test (factorial 5 = 120);" in result
        assert "__test (factorial 0 = 1);" in result
        assert "GRPO_TEST_RESULT" in result

    def test_trailing_semicolon_preserved(self):
        """Trailing semicolons after last assertion are handled."""
        tests = """let () =
        assert (x = 1);
        assert (y = 2);;"""
        result = transform_tests_for_partial_credit(tests)

        # Should end with valid OCaml syntax
        assert result.rstrip().endswith(";;")


# ============================================================================
# parse_test_results Tests
# ============================================================================


class TestParseTestResults:
    """Tests for parsing GRPO_TEST_RESULT from stdout."""

    def test_parse_valid_result(self):
        """Valid GRPO_TEST_RESULT is parsed correctly."""
        stdout = "GRPO_TEST_RESULT:5/7\n"
        passed, total = parse_test_results(stdout)

        assert passed == 5
        assert total == 7

    def test_parse_all_passed(self):
        """All tests passing is parsed correctly."""
        stdout = "GRPO_TEST_RESULT:10/10\n"
        passed, total = parse_test_results(stdout)

        assert passed == 10
        assert total == 10

    def test_parse_none_passed(self):
        """No tests passing is parsed correctly."""
        stdout = "GRPO_TEST_RESULT:0/5\n"
        passed, total = parse_test_results(stdout)

        assert passed == 0
        assert total == 5

    def test_parse_with_other_output(self):
        """Result embedded in other output is found."""
        stdout = "Starting tests...\nGRPO_TEST_RESULT:3/4\nDone.\n"
        passed, total = parse_test_results(stdout)

        assert passed == 3
        assert total == 4

    def test_parse_missing_result(self):
        """Missing result returns (0, 0)."""
        stdout = "No test results here\n"
        passed, total = parse_test_results(stdout)

        assert passed == 0
        assert total == 0

    def test_parse_empty_stdout(self):
        """Empty stdout returns (0, 0)."""
        passed, total = parse_test_results("")

        assert passed == 0
        assert total == 0

    def test_parse_malformed_result(self):
        """Malformed result returns (0, 0)."""
        stdout = "GRPO_TEST_RESULT:abc/def\n"
        passed, total = parse_test_results(stdout)

        assert passed == 0
        assert total == 0


# ============================================================================
# ASSERT_PATTERN Regex Tests
# ============================================================================


class TestAssertPattern:
    """Tests for the ASSERT_PATTERN regex."""

    def test_simple_assertion(self):
        """Simple assertion is matched."""
        text = "assert (x = 1);"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 1
        assert matches[0].group(1) == "(x = 1)"

    def test_assertion_with_whitespace(self):
        """Assertion with various whitespace is matched."""
        text = "assert   (  x = 1  )  ;"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 1

    def test_multiple_assertions(self):
        """Multiple assertions are all matched."""
        text = "assert (a = 1); assert (b = 2); assert (c = 3);"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 3

    def test_nested_parens_matched(self):
        """Nested parentheses are matched correctly."""
        text = "assert (foo (bar (baz x)) = 42);"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 1
        assert "(foo (bar (baz x)) = 42)" in matches[0].group(1)

    def test_list_with_semicolons(self):
        """List literals with semicolons are matched correctly."""
        text = "assert (lst = [1; 2; 3]);"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 1
        assert "[1; 2; 3]" in matches[0].group(1)

    def test_nested_list_with_semicolons(self):
        """Nested lists with semicolons are matched correctly."""
        text = "assert (matrix = [[0; 0]; [1; 0]]);"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 1
        assert "[[0; 0]; [1; 0]]" in matches[0].group(1)

    def test_no_match_without_semicolon(self):
        """Assertion without trailing semicolon is not matched."""
        text = "assert (x = 1)"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 0

    def test_no_match_without_parens(self):
        """Assertion without parentheses is not matched."""
        text = "assert x = 1;"
        matches = list(ASSERT_PATTERN.finditer(text))

        assert len(matches) == 0


# ============================================================================
# Graduated Test Scoring Tests
# ============================================================================


@pytest.mark.skipif(not shutil.which("ocamlc"), reason="OCaml not installed")
class TestGraduatedScoring:
    """Tests for graduated test scoring based on pass rate."""

    def test_full_pass_gets_full_score(self):
        """All tests passing gets TESTS_PASS_SCORE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Code that prints 5/5 passed
            source.write_text("""
            let () = Printf.printf "GRPO_TEST_RESULT:5/5\\n"
            """)
            # Compile
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            assert result.score == TESTS_PASS_SCORE
            assert result.metadata["tests_passed"] == 5
            assert result.metadata["tests_total"] == 5
            assert result.metadata["pass_rate"] == 1.0

    def test_partial_pass_gets_proportional_score(self):
        """Partial pass gets proportional score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Code that prints 3/5 passed
            source.write_text("""
            let () = Printf.printf "GRPO_TEST_RESULT:3/5\\n"
            """)
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            expected_score = TESTS_PASS_SCORE * (3 / 5)  # 0.65 * 0.6 = 0.39
            assert result.score == pytest.approx(expected_score, abs=0.001)
            assert result.metadata["tests_passed"] == 3
            assert result.metadata["tests_total"] == 5
            assert result.metadata["pass_rate"] == pytest.approx(0.6, abs=0.001)

    def test_no_pass_gets_zero(self):
        """Zero tests passing gets zero score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Code that prints 0/5 passed
            source.write_text("""
            let () = Printf.printf "GRPO_TEST_RESULT:0/5\\n"
            """)
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            assert result.score == 0.0
            assert result.metadata["tests_passed"] == 0
            assert result.metadata["tests_total"] == 5
            assert result.metadata["pass_rate"] == 0.0

    def test_fallback_to_exit_code_success(self):
        """No GRPO_TEST_RESULT with exit 0 gets full score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Code that exits successfully without printing result
            source.write_text("let () = ()")
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            assert result.score == TESTS_PASS_SCORE
            assert result.metadata["tests_passed"] == 1
            assert result.metadata["tests_total"] == 1

    def test_fallback_to_exit_code_failure(self):
        """No GRPO_TEST_RESULT with exit 1 gets zero score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Code that exits with failure
            source.write_text("let () = exit 1")
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            assert result.score == 0.0


# ============================================================================
# End-to-End Partial Credit Tests
# ============================================================================


@pytest.mark.skipif(not shutil.which("ocamlc"), reason="OCaml not installed")
class TestEndToEndPartialCredit:
    """End-to-end tests verifying full partial credit pipeline."""

    def test_transformed_tests_compile(self):
        """Transformed test code compiles successfully."""
        tests = """let () =
        assert (1 + 1 = 2);
        assert (2 * 2 = 4);
        assert (3 - 1 = 2);;"""

        transformed = transform_tests_for_partial_credit(tests)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text(transformed)

            result = subprocess.run(
                ["ocamlc", "-c", str(source)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Compile failed: {result.stderr}"

    def test_transformed_tests_run_and_report(self):
        """Transformed tests run and produce parseable output."""
        tests = """let () =
        assert (1 + 1 = 2);
        assert (2 * 2 = 4);
        assert (3 - 1 = 2);;"""

        transformed = transform_tests_for_partial_credit(tests)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text(transformed)

            # Compile
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            # Run
            result = subprocess.run(
                ["./runner"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            passed, total = parse_test_results(result.stdout)

            assert total == 3
            assert passed == 3

    def test_partial_pass_with_real_code(self):
        """Real OCaml code with partial test pass produces correct reward."""
        # Solution that only works for some inputs
        solution = """let double x = x + x"""

        tests = """let () =
        assert (double 1 = 2);
        assert (double 2 = 4);
        assert (double 0 = 0);
        assert (double 5 = 10);;"""

        transformed = transform_tests_for_partial_credit(tests)
        combined = solution + "\n\n" + transformed

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text(combined)

            # Compile
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            # All 4 tests should pass (double works correctly)
            assert result.score == TESTS_PASS_SCORE
            assert result.metadata["tests_passed"] == 4
            assert result.metadata["tests_total"] == 4

    def test_failing_tests_produce_partial_score(self):
        """Failing tests produce proportional partial score."""
        # Solution that fails for some inputs
        solution = """let is_positive x = x > 0"""

        tests = """let () =
        assert (is_positive 1 = true);
        assert (is_positive (-1) = false);
        assert (is_positive 0 = true);;"""  # This one fails (0 > 0 is false)

        transformed = transform_tests_for_partial_credit(tests)
        combined = solution + "\n\n" + transformed

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text(combined)

            # Compile
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            # 2/3 tests should pass
            assert result.metadata["tests_passed"] == 2
            assert result.metadata["tests_total"] == 3
            expected_score = TESTS_PASS_SCORE * (2 / 3)
            assert result.score == pytest.approx(expected_score, abs=0.001)

    def test_complex_assertions_work_end_to_end(self):
        """Complex assertions with nested structures work correctly."""
        solution = """let sum_list lst = List.fold_left (+) 0 lst"""

        tests = """let () =
        assert (sum_list [1; 2; 3] = 6);
        assert (sum_list [] = 0);
        assert (sum_list [10; 20; 30; 40] = 100);;"""

        transformed = transform_tests_for_partial_credit(tests)
        combined = solution + "\n\n" + transformed

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text(combined)

            # Compile
            subprocess.run(
                ["ocamlc", "-o", "runner", str(source)],
                cwd=tmpdir,
                check=True,
            )

            result = tests_reward(tmpdir, "runner")

            assert result.metadata["tests_passed"] == 3
            assert result.metadata["tests_total"] == 3
            assert result.score == TESTS_PASS_SCORE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
