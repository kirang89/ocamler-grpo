#!/usr/bin/env python3
"""
Unit tests for reward.py parallel execution.

Verifies that parallel reward computation produces identical results
to sequential execution and preserves result ordering.
"""

import unittest

from reward import _compute_rewards_parallel, _score_completion_vf


class TestRewardParallelization(unittest.TestCase):
    """Test that parallel reward computation matches sequential behavior."""

    VALID_OCAML = """```ocaml
let sum a b = a + b
let () = assert (sum 1 2 = 3)
```"""

    INVALID_OCAML = """```ocaml
let sum a b = a +
```"""

    EMPTY_COMPLETION = ""

    PROSE_COMPLETION = """Here's how to solve this problem:
```ocaml
let x = 1
```
This approach works because...
"""

    SIMPLE_TEST = "let () = ()"

    def test_parallel_matches_sequential_multiple(self):
        """Multiple completions should produce same results in parallel."""
        completions = [
            self.VALID_OCAML,
            self.INVALID_OCAML,
            self.EMPTY_COMPLETION,
            self.PROSE_COMPLETION,
        ]
        ids = ["test_1", "test_2", "test_3", "test_4"]
        tests = [self.SIMPLE_TEST] * 4

        parallel_results = _compute_rewards_parallel(completions, ids, tests)

        sequential_results = [
            _score_completion_vf(ids[i], completions[i], tests[i])
            for i in range(len(completions))
        ]

        self.assertEqual(len(parallel_results), len(sequential_results))

        for i in range(len(completions)):
            self.assertEqual(
                parallel_results[i]["total_reward"],
                sequential_results[i]["total_reward"],
                f"Mismatch at index {i}",
            )
            self.assertEqual(
                parallel_results[i]["problem_id"],
                sequential_results[i]["problem_id"],
                f"Problem ID mismatch at index {i}",
            )

    def test_result_ordering_preserved(self):
        """Results should be in same order as input completions."""
        completions = [self.VALID_OCAML, self.EMPTY_COMPLETION, self.INVALID_OCAML]
        ids = ["first", "second", "third"]
        tests = [self.SIMPLE_TEST] * 3

        results = _compute_rewards_parallel(completions, ids, tests)

        self.assertEqual(results[0]["problem_id"], "first")
        self.assertEqual(results[1]["problem_id"], "second")
        self.assertEqual(results[2]["problem_id"], "third")


if __name__ == "__main__":
    unittest.main()
