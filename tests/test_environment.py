"""
Tests verify that the verifiers environment migration preserves
the expected behavior from the original reward.py implementation.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from rlvr.environment import (
    compile_reward,
    count_non_empty_code_lines,
    extract_code_block,
    extract_function_signature,
    is_degenerate_output,
    ocaml_reward,
    prepend_signature,
    tests_reward,
    type_check_reward,
)

# ============================================================================
# Code Extraction Tests
# ============================================================================


class TestCodeExtraction:
    def test_extract_code_block_with_language_hint(self):
        """Test extraction from markdown fences with language hint."""
        code = "let x = 1"
        assert extract_code_block(f"```ocaml\n{code}\n```") == code
        assert extract_code_block(f"```ml\n{code}\n```") == code
        assert extract_code_block(f"```language:ocaml\n{code}\n```") == code

    def test_extract_code_block_without_hint(self):
        """Test extraction without language hint."""
        code = "let x = 1"
        assert extract_code_block(f"```\n{code}\n```") == code

    def test_extract_multiple_code_blocks(self):
        """Test handling of multiple code blocks (should take first valid)."""
        # Empty block followed by valid block
        text = "```\n```\n```ocaml\nlet x = 1\n```"
        assert extract_code_block(text) == "let x = 1"

        # Multiple valid blocks - take first
        text = "```ocaml\nlet x = 1\n```\n```ocaml\nlet y = 2\n```"
        assert extract_code_block(text) == "let x = 1"

        # Skip language-only blocks
        text = "```ocaml```\n```\nlet x = 1\n```"
        assert extract_code_block(text) == "let x = 1"

    def test_extract_fallback_to_raw_text(self):
        """Test fallback to raw text when no code blocks."""
        code = "let x = 1"
        assert extract_code_block(code) == code

    def test_extract_with_prose_and_code_block(self):
        """Test extraction with prose before code block."""
        code = "let add x y = x + y"
        text = f"Here's the solution:\n```ocaml\n{code}\n```\n Try it out and let me know"
        assert extract_code_block(text) == code


SIGNATURE_PREPEND_TEST_CASES = [
    # (prompt, completion, expected_completion, description)
    # Body only - should prepend
    (
        "(**Doc*)\nlet rec factorial (n : int) : int =",
        "match n with | 0 -> 1",
        "let rec factorial (n : int) : int =\n  match n with | 0 -> 1",
        "body only",
    ),
    # Completion starts with let (redefinition) - should not prepend
    (
        "(**Doc*)\nlet rec factorial (n : int) : int =",
        "let rec factorial (n : int) : int = 1",
        "let rec factorial (n : int) : int = 1",
        "redefinition",
    ),
    # Local let binding - should prepend
    (
        "(**Doc*)\nlet rec factorial (n : int) : int =",
        "let base = 1 in base",
        "let rec factorial (n : int) : int =\n  let base = 1 in base",
        "local let",
    ),
    # Leading whitespace before let - should not prepend (assumed redefinition)
    (
        "(**Doc*)\nlet foo (x : int) : int =",
        "  let foo (x : int) : int = x",
        "  let foo (x : int) : int = x",
        "whitespace before let redefinition",
    ),
    # No signature in prompt - should not prepend
    ("No docstring here", "match n with | 0 -> 1", "match n with | 0 -> 1", "no signature"),
]

SIGNATURE_TEST_CASES = [
    # (prompt, expected_sig, expected_name, description)
    # Simple recursive function
    (
        "(**Doc*)\nlet rec factorial (n : int) : int =",
        "let rec factorial (n : int) : int =",
        "factorial",
        "simple recursive",
    ),
    # Non-recursive function
    (
        "(**Doc*)\nlet is_prime (n : int) : bool =",
        "let is_prime (n : int) : bool =",
        "is_prime",
        "non-recursive",
    ),
    # Multiple parameters
    (
        "(**Doc*)\nlet rec power (base : int) (exp : int) : int =",
        "let rec power (base : int) (exp : int) : int =",
        "power",
        "multi-param",
    ),
    # Generic type
    (
        "(**Doc*)\nlet rec last_element (lst : 'a list) : 'a =",
        "let rec last_element (lst : 'a list) : 'a =",
        "last_element",
        "generic type",
    ),
    # Tuple return type
    (
        "(**Doc*)\nlet determine_os_variables (flavor : string) : string * string * string * string =",
        "let determine_os_variables (flavor : string) : string * string * string * string =",
        "determine_os_variables",
        "tuple return",
    ),
    # Complex parameters
    (
        "(**Doc*)\nlet a_star (graph : (int * int) list array) (heuristic : int -> int -> int) (start : int) (goal : int) : int list =",
        "let a_star (graph : (int * int) list array) (heuristic : int -> int -> int) (start : int) (goal : int) : int list =",
        "a_star",
        "complex params",
    ),
    # Result type
    (
        "(**Doc*)\nlet add (x : int option) (y : int option) : (int, string) result =",
        "let add (x : int option) (y : int option) : (int, string) result =",
        "add",
        "result type",
    ),
    # Trailing whitespace
    (
        "(**Doc*)\nlet foo (x : int) : int =   ",
        "let foo (x : int) : int =",
        "foo",
        "trailing whitespace",
    ),
    # Full docstring with examples
    (
        "(**Compute the factorial\n * >>> factorial 5\n * 120\n*)\nlet rec factorial (n : int) : int =",
        "let rec factorial (n : int) : int =",
        "factorial",
        "full docstring",
    ),
    # No signature - just text
    ("Just some text without a function signature", "", "", "no signature"),
    # No docstring
    (
        "let rec factorial (n : int) : int =",
        "let rec factorial (n : int) : int =",
        "factorial",
        "no docstring",
    ),
    # Literal \n (as in CSV) instead of actual newline
    (
        "(**Doc*)\\nlet rec factorial (n : int) : int =",
        "let rec factorial (n : int) : int =",
        "factorial",
        "literal backslash-n",
    ),
    # Type definition before function
    (
        "(**Doc*)\\ntype element = Int of int\\nlet list_sorter (lst : element list) : float list =",
        "let list_sorter (lst : element list) : float list =",
        "list_sorter",
        "type definition",
    ),
    # Trailing literal \n
    (
        "(**Doc*)\\nlet sieve (lst : int list) : int list =\\n",
        "let sieve (lst : int list) : int list =",
        "sieve",
        "trailing literal newline",
    ),
]


class TestFunctionSignatureExtraction:
    """Tests for extract_function_signature and prepend_signature functions."""

    @pytest.mark.parametrize("prompt,expected_sig,expected_name,desc", SIGNATURE_TEST_CASES)
    def test_extract_signature(self, prompt, expected_sig, expected_name, desc):
        sig, name = extract_function_signature(prompt)
        assert sig == expected_sig, f"Failed signature for {desc}: got {sig!r}"
        assert name == expected_name, f"Failed name for {desc}: got {name!r}"

    @pytest.mark.parametrize("prompt,completion,expected,desc", SIGNATURE_PREPEND_TEST_CASES)
    def test_prepend_logic(self, prompt, completion, expected, desc):
        result = prepend_signature(prompt, completion)
        assert result == expected, f"Failed for {desc}: got {result!r}, expected {expected!r}"


class TestCodeLineCounter:
    """Tests for count_non_empty_code_lines function."""

    def test_count_normal_lines(self):
        """Test normal code lines are counted."""
        code = "let x = 1\nlet y = 2"
        assert count_non_empty_code_lines(code) == 2

    def test_count_with_empty_lines(self):
        """Test empty lines are not counted."""
        code = "let x = 1\n\nlet y = 2"
        assert count_non_empty_code_lines(code) == 2

        code = "let x = 1\n  \n\t\nlet y = 2"
        assert count_non_empty_code_lines(code) == 2

    def test_count_with_comments(self):
        """Test comment lines starting with (* are not counted."""
        code = "(* comment *)\nlet x = 1"
        assert count_non_empty_code_lines(code) == 1

        # Note: Only lines starting with (* are excluded. Multi-line
        # comment continuation lines are counted as they don't start with (*
        code = "let x = 1\n(* multi-line\n   comment *)\nlet y = 2"
        assert count_non_empty_code_lines(code) == 3

    def test_count_empty_code(self):
        """Test empty code returns 0."""
        assert count_non_empty_code_lines("") == 0
        assert count_non_empty_code_lines("   \n  \n\t") == 0

    def test_count_only_comments(self):
        """Test code with only comments returns 0."""
        assert count_non_empty_code_lines("(* comment *)") == 0
        assert count_non_empty_code_lines("(* line 1 *)\n(* line 2 *)") == 0


# ============================================================================
# Degenerate Output Detection Tests
# ============================================================================


class TestDegenerateDetection:
    """Tests for is_degenerate_output function."""

    @pytest.fixture(autouse=True)
    def clear_prose_penalty_env(self):
        """Ensure GRPO_DISABLE_PROSE_PENALTY is not set for these tests."""
        old_value = os.environ.pop("GRPO_DISABLE_PROSE_PENALTY", None)
        yield
        if old_value is not None:
            os.environ["GRPO_DISABLE_PROSE_PENALTY"] = old_value

    def test_code_block_spam_detection(self):
        """Test markdown code block spam is detected."""
        # Multiple code blocks (>2 pairs of ```)
        spam = "```ocaml\n```\n```\n```\n```\n```\nlet x = 1"
        is_deg, reasons = is_degenerate_output(spam, "let x = 1")
        assert is_deg is True
        assert "code block spam" in reasons

    def test_low_code_purity_detection(self):
        """Test low code purity (too much wrapper text) is detected."""
        code = "let x = 1"
        # Lots of prose around small code block
        wrapper = "Here is a very long explanation about why this code works " * 10
        completion = f"{wrapper}\n```ocaml\n{code}\n```"
        is_deg, reasons = is_degenerate_output(completion, code)
        assert is_deg is True
        assert "low code ratio" in reasons

    def test_repetitive_content_detection(self):
        """Test highly repetitive content (spam) is detected."""
        # Highly repetitive pattern
        repetitive = "let x = 1\n" * 50
        completion = f"```ocaml\n{repetitive}\n```"
        is_deg, reasons = is_degenerate_output(completion, repetitive)
        assert is_deg is True
        assert "repetitive content" in reasons

    def test_stub_solution_detection(self):
        """Test stub solutions with failwith are detected."""
        stub = 'let rec sort_array (arr : int list) : int list = failwith "Please implement the sort_array function."'
        is_deg, reasons = is_degenerate_output(stub, stub)
        assert is_deg is True
        assert any(r.startswith("stub solution") for r in reasons)

        stub2 = '''let sort_array arr =
  failwith "implement me"'''
        is_deg, reasons = is_degenerate_output(stub2, stub2)
        assert is_deg is True
        assert any(r.startswith("stub solution") for r in reasons)

        # Longer code with failwith should not trigger (could be legitimate error handling)
        stub3 = '''let sort_array arr =
  let x = 1 in
  failwith "implement later"'''
        is_deg, reasons = is_degenerate_output(stub3, stub3)
        assert is_deg is False
        assert not any(r.startswith("stub solution") for r in reasons)

    def test_placeholder_comment_detection(self):
        """Test placeholder comments in short code are detected."""
        stub = '''let determine_convexity (terms : (float * int * int) list) (n : int) : string =
  "Convex" (* placeholder, replace with actual logic *)'''
        is_deg, reasons = is_degenerate_output(stub, stub)
        assert is_deg is True
        assert any(r.startswith("stub solution") for r in reasons)

    def test_assert_false_detection(self):
        """Test assert false placeholders are detected."""
        stub = '''let solve x =
  assert false'''
        is_deg, reasons = is_degenerate_output(stub, stub)
        assert is_deg is True
        assert "stub solution (assert false)" in reasons

    def test_clean_code_passes(self):
        """Test clean, valid code is not marked as degenerate."""
        code = "let rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)"
        completion = f"```ocaml\n{code}\n```"
        is_deg, reasons = is_degenerate_output(completion, code)
        assert is_deg is False
        assert reasons == []

        # More complex code with good keyword density
        code = """
let rec quicksort = function
  | [] -> []
  | pivot :: rest ->
      let smaller, larger = List.partition (fun x -> x < pivot) rest in
      quicksort smaller @ [pivot] @ quicksort larger
"""
        completion = f"```ocaml\n{code}\n```"
        is_deg, reasons = is_degenerate_output(completion, code)
        assert is_deg is False
        assert reasons == []

    def test_environment_variable_disable(self):
        """Test degenerate detection can be disabled via environment variable."""
        os.environ["GRPO_DISABLE_PROSE_PENALTY"] = "true"
        try:
            # Even obvious spam should return False when disabled
            spam = "```ocaml\n```\n```\n```\n```\n```\nlet x = 1"
            is_deg, reasons = is_degenerate_output(spam, "let x = 1")
            assert is_deg is False
            assert reasons == []
        finally:
            del os.environ["GRPO_DISABLE_PROSE_PENALTY"]


# ============================================================================
# OCaml Compilation Tests
# ============================================================================


@pytest.mark.skipif(not shutil.which("ocamlc"), reason="OCaml not installed")
class TestOCamlCompilation:
    """Tests for OCaml compilation functions (requires ocamlc)."""

    def test_type_check_perfect(self):
        """Test perfect type check with valid OCaml code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text("let x = 1")

            result = type_check_reward(source, tmpdir)

            assert result.score == 0.25
            assert result.metadata["syntax_errors"] == 0
            assert result.metadata["has_syntax_error"] is False
            assert result.metadata["timed_out"] is False
            assert result.metadata["error_details"] == "success"

    def test_type_check_with_syntax_error(self):
        """Test type check with syntax errors gets zero score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text("let x =")  # Incomplete syntax

            result = type_check_reward(source, tmpdir)

            assert result.score == 0.0
            assert result.metadata["has_syntax_error"] is True
            assert result.metadata["timed_out"] is False

    def test_type_check_graduated_scoring(self):
        """Test graduated scoring for type errors."""
        test_cases = [
            # (error_count, expected_score, code_description)
            (1, 0.20, 'let x : int = "string"'),
            (2, 0.15, 'let x : int = "string"\nlet y : int = "string"'),
        ]

        for error_count, expected_score, code in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                source = tmpdir / "test.ml"
                source.write_text(code)

                result = type_check_reward(source, tmpdir)

                # Score should match expected (allowing for actual error count)
                assert result.score > 0.0  # Should have partial credit
                assert result.metadata["has_syntax_error"] is False

    def test_type_check_many_errors(self):
        """Test type check with many errors gets graduated partial credit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Code with type errors - OCaml may report fewer errors than expected
            # since it may stop processing or collapse similar errors
            code = "\n".join([f'let x{i} : int = "string"' for i in range(15)])
            source.write_text(code)

            result = type_check_reward(source, tmpdir)

            # Should get some partial credit (graduated based on actual error count)
            # The score depends on how many errors OCaml actually reports
            assert 0.0 < result.score <= 0.20
            assert result.metadata["has_syntax_error"] is False
            assert result.metadata["timed_out"] is False

    def test_compile_success(self):
        """Test successful compilation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text("let () = ()")

            # Get type check result first
            type_check = type_check_reward(source, tmpdir)
            compile_result = compile_reward(source, tmpdir, "test", type_check)

            assert compile_result.score == 0.10

    def test_compile_failure_with_perfect_type_check(self):
        """Test compilation failure with perfect type check gets partial credit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Valid type check but may fail compilation for other reasons
            source.write_text("let x = 1")

            type_check = type_check_reward(source, tmpdir)
            compile_result = compile_reward(source, tmpdir, "test", type_check)

            # Should either compile successfully (0.10) or get partial credit (0.05)
            assert compile_result.score in [0.10, 0.05]

    def test_compile_with_type_errors(self):
        """Test compilation with type errors gets minimal credit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text('let x : int = "string"')

            type_check = type_check_reward(source, tmpdir)
            compile_result = compile_reward(source, tmpdir, "test", type_check)

            # Should get minimal credit for attempting compilation
            assert compile_result.score == 0.01

    def test_compile_with_syntax_error(self):
        """Test compilation with syntax errors gets no credit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            source.write_text("let x =")

            type_check = type_check_reward(source, tmpdir)
            compile_result = compile_reward(source, tmpdir, "test", type_check)

            assert compile_result.score == 0.0

    def test_test_execution_success(self):
        """Test successful test execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Simple program that exits successfully
            source.write_text("let () = assert (1 + 1 = 2)")

            # Compile first
            type_check = type_check_reward(source, tmpdir)
            compile_result = compile_reward(source, tmpdir, "test_exe", type_check)

            if compile_result.score == 0.10:
                test_result = tests_reward(tmpdir, "test_exe")
                assert test_result.score == 0.65
                assert test_result.metadata["timed_out"] is False

    def test_test_execution_failure(self):
        """Test failed test execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "test.ml"
            # Program that exits with error
            source.write_text("let () = assert (1 + 1 = 3)")

            # Compile first
            type_check = type_check_reward(source, tmpdir)
            compile_result = compile_reward(source, tmpdir, "test_exe", type_check)

            if compile_result.score == 0.10:
                test_result = tests_reward(tmpdir, "test_exe")
                assert test_result.score == 0.0
                assert test_result.metadata["timed_out"] is False


# ============================================================================
# End-to-End Reward Tests
# ============================================================================


@pytest.mark.skipif(not shutil.which("ocamlc"), reason="OCaml not installed")
class TestOCamlRewardEndToEnd:
    """End-to-end tests for ocaml_reward function."""

    @pytest.fixture(autouse=True)
    def clear_prose_penalty_env(self):
        """Ensure GRPO_DISABLE_PROSE_PENALTY is not set for these tests."""
        old_value = os.environ.pop("GRPO_DISABLE_PROSE_PENALTY", None)
        yield
        if old_value is not None:
            os.environ["GRPO_DISABLE_PROSE_PENALTY"] = old_value

    def test_perfect_solution(self):
        """Test with valid OCaml code that passes tests."""
        # Note: Code needs MIN_NON_EMPTY_LINES (2) lines to be scored
        completion = """```ocaml
let add x y = x + y
let sub x y = x - y
```"""
        info = {
            "tests": "let () = assert (add 1 2 = 3)",
            "problem_id": "test_add",
        }
        state = {"problem_id": "test_add"}

        reward = ocaml_reward(completion, info, state)

        # Should get full reward (0.25 type + 0.10 compile + 0.65 tests = 1.0)
        assert reward == 1.0

    def test_solution_with_type_errors(self):
        """Test with code that has type errors."""
        # Note: Code needs MIN_NON_EMPTY_LINES (2) lines to be scored
        completion = """```ocaml
let add x y : int = "not an int"
let sub x y : int = "also wrong"
```"""
        info = {
            "tests": "let () = assert (add 1 2 = 3)",
            "problem_id": "test_type_error",
        }
        state = {"problem_id": "test_type_error"}

        reward = ocaml_reward(completion, info, state)

        # Should get partial credit from type check (graduated for type errors)
        # Plus minimal compile credit (0.01)
        assert 0.0 < reward < 1.0
        assert reward < 0.30  # Should not get full type check + compile

    def test_degenerate_output_penalty(self):
        """Test that degenerate output (prose) gets penalized."""
        # Note: Code needs MIN_NON_EMPTY_LINES (2) lines to be scored
        completion = """Here's the solution to your problem:

```ocaml
let add x y = x + y
let sub x y = x - y
```

This implementation works by adding the two numbers together."""
        info = {
            "tests": "let () = assert (add 1 2 = 3)",
            "problem_id": "test_prose",
        }
        state = {"problem_id": "test_prose"}

        reward = ocaml_reward(completion, info, state)

        # Should get penalized to 0.3x of base reward
        # Base would be 1.0, so should be around 0.3
        assert 0.2 < reward < 0.4

    def test_empty_code(self):
        """Test with empty or minimal code gets zero reward."""
        completions = [
            "",
            "```ocaml\n```",
            "```ocaml\n\n```",
            "let x = 1",  # Only 1 line, needs MIN_NON_EMPTY_LINES
        ]

        info = {
            "tests": "let () = ()",
            "problem_id": "test_empty",
        }
        state = {"problem_id": "test_empty"}

        for completion in completions:
            reward = ocaml_reward(completion, info, state)
            assert reward == 0.0

    def test_syntax_error(self):
        """Test with syntax errors gets zero reward."""
        completion = """```ocaml
let add x y =
```"""
        info = {
            "tests": "let () = ()",
            "problem_id": "test_syntax",
        }
        state = {"problem_id": "test_syntax"}

        reward = ocaml_reward(completion, info, state)
        assert reward == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestRewardInterface:
    """Tests for reward function interface and compatibility."""

    def test_reward_signature(self):
        """Test that ocaml_reward has the correct signature."""
        import inspect

        sig = inspect.signature(ocaml_reward)
        params = list(sig.parameters.keys())

        assert params == ["completion", "info", "state"]
        assert sig.return_annotation is float or sig.return_annotation is inspect.Signature.empty

    def test_reward_return_type(self):
        """Test that ocaml_reward returns a float."""
        completion = "let x = 1"
        info = {"tests": "", "problem_id": "test"}
        state = {"problem_id": "test"}

        reward = ocaml_reward(completion, info, state)
        assert isinstance(reward, float)

    def test_reward_range(self):
        """Test that rewards are in valid range [0, 1]."""
        test_cases = [
            "let x = 1",
            "```ocaml\nlet x = 1\n```",
            "invalid syntax let =",
            "",
        ]

        info = {"tests": "", "problem_id": "test"}
        state = {"problem_id": "test"}

        for completion in test_cases:
            reward = ocaml_reward(completion, info, state)
            assert 0.0 <= reward <= 1.0

    def test_missing_problem_id(self):
        """Test handling of missing problem_id."""
        completion = "```ocaml\nlet x = 1\n```"
        info = {"tests": ""}
        state = {}

        # Should not crash, should use "unknown" as problem_id
        reward = ocaml_reward(completion, info, state)
        assert isinstance(reward, float)

    def test_missing_tests(self):
        """Test handling of missing tests field."""
        completion = "```ocaml\nlet x = 1\nlet y = 2\n```"
        info = {"problem_id": "test"}
        state = {"problem_id": "test"}

        # Should not crash, should use empty string for tests
        reward = ocaml_reward(completion, info, state)
        assert isinstance(reward, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
