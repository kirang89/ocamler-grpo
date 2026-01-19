"""Tests for solution style penalty."""

import pytest

from rlvr.environment import CODE_BLOCK_RE
from rlvr.reward import compute_solution_style_penalty


# Dummy code (not used for style check, just need a non-empty string)
CODE = "let rec fibonacci n = match n with | 0 -> 0 | 1 -> 1 | _ -> fibonacci (n-1) + fibonacci (n-2)"

# Solution #1 - multiple blocks + trailing prose
VERBOSE_SOLUTION = """
Solution:
<code>
let rec fibonacci (n : int) : int =
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)
</code>

<code>
</code>
Here is the solution based on the provided problem:

<code>
let fibonacci (n : int) : int =
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)
</code>

This function uses a recursive approach to calculate the Fibonacci number of a given index `n`. The base cases are when `n` is 0 or 1, in which case the function returns 0 and 1 respectively. For other values of `n`, the function recursively calls itself with `n-1` and `n-2`, and then adds the results together to get the Fibonacci number. The solution covers all the provided examples by correctly implementing the Fibonacci calculation based on the recursive definition."""

# Solution #2 - clean
CLEAN_SOLUTION = """
Solution:
<code>
let rec fibonacci (n : int) : int =
  match n with
  | 0 -> 0
  | 1 -> 1
  | _ -> fibonacci (n - 1) + fibonacci (n - 2)
</code>"""

# Solution #3 - very verbose with multiple blocks + trailing prose
VERY_VERBOSE_SOLUTION = """
Solution:
<code>
let rec fibonacci (n : int) : int =
  match n with
  | 0 -> 0
  | 1 -> 1
  | _ -> fibonacci (n - 1) + fibonacci (n - 2)
</code>

<code>
</code> Solution:
<code>
let fibonacci (n : int) : int =
  match n with
  | 0 -> 0
  | 1 -> 1
  | _ -> fibonacci (n - 1) + fibonacci (n - 2)
</code>

Given the sequence defined by:
a_n = 0 if n = 0
      1 if n = 1
      a_{n-1} + a_{n-2} if n > 1

Write a function `fibonacci` in OCaml to compute `a_n`.

Here's how you can implement the `fibonacci` function in OCaml:

<code>
let rec fibonacci (n : int) : int =
  match n with
  | 0 -> 0
  | 1 -> 1
  | _ -> fibonacci (n - 1) + fibonacci (n - 2)

let () =
  Printf.printf "fibonacci 0 = %d\\n" (fibonacci 0);
  Printf.printf "fibonacci 1 = %d\\n" (fibonacci 1);
  Printf.printf "fibonacci 5 = %d\\n" (fibonacci 5)
</code>

Explanation: The function uses recursive pattern matching."""


class TestStylePenalty:
    """Tests for compute_solution_style_penalty."""

    def test_clean_solution_no_penalty(self):
        """Clean solution with single code block gets no penalty."""
        penalty, reasons = compute_solution_style_penalty(CLEAN_SOLUTION, CODE, CODE_BLOCK_RE)
        assert penalty == 0.0
        assert reasons == []

    def test_verbose_solution_penalized(self):
        """Verbose solution with multiple blocks and prose gets penalized."""
        penalty, reasons = compute_solution_style_penalty(VERBOSE_SOLUTION, CODE, CODE_BLOCK_RE)
        assert penalty > 0.0
        assert "code blocks" in reasons[0]
        assert "trailing prose" in reasons[1]

    def test_very_verbose_solution_penalized(self):
        """Very verbose solution gets higher penalty."""
        penalty, reasons = compute_solution_style_penalty(VERY_VERBOSE_SOLUTION, CODE, CODE_BLOCK_RE)
        assert penalty > 0.0
        assert "code blocks" in reasons[0]
        assert "trailing prose" in reasons[1]

    def test_penalty_capped_at_010(self):
        """Penalty should never exceed 0.10."""
        penalty, _ = compute_solution_style_penalty(VERY_VERBOSE_SOLUTION, CODE, CODE_BLOCK_RE)
        assert penalty <= 0.10

    def test_clean_beats_verbose(self):
        """Clean solution should have lower penalty than verbose."""
        clean_penalty, _ = compute_solution_style_penalty(CLEAN_SOLUTION, CODE, CODE_BLOCK_RE)
        verbose_penalty, _ = compute_solution_style_penalty(VERBOSE_SOLUTION, CODE, CODE_BLOCK_RE)
        assert clean_penalty < verbose_penalty

    def test_multiple_code_blocks_penalty(self):
        """Each extra code block adds 0.02 penalty."""
        # 3 code blocks = 2 extra = 0.04 penalty
        completion = "<code>code</code>\n<code>code</code>\n<code>code</code>"
        penalty, reasons = compute_solution_style_penalty(completion, CODE, CODE_BLOCK_RE)
        assert penalty == pytest.approx(0.04)
        assert "3 code blocks" in reasons[0]

    def test_trailing_prose_penalty(self):
        """Trailing prose after code block adds 0.03 penalty."""
        completion = "<code>code</code>\n\nThis is a long explanation that exceeds 30 characters."
        penalty, reasons = compute_solution_style_penalty(completion, CODE, CODE_BLOCK_RE)
        assert penalty == pytest.approx(0.03)
        assert "trailing prose" in reasons[0]

    def test_short_trailing_content_no_penalty(self):
        """Short trailing content (<=30 chars) doesn't trigger penalty."""
        completion = "<code>code</code>\nshort"
        penalty, reasons = compute_solution_style_penalty(completion, CODE, CODE_BLOCK_RE)
        assert penalty == 0.0
        assert reasons == []
