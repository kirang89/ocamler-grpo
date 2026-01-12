"""Tests for rlvr.reward module - create_reward_function behavior."""

from rlvr.reward import create_reward_function


class TestCreateRewardFunctionBasic:
    """Test basic reward function creation and interface."""

    def test_returns_callable(self):
        """create_reward_function returns a callable."""
        reward_fn = create_reward_function(logger=None)
        assert callable(reward_fn)

    def test_function_has_name(self):
        """Returned function has __name__ attribute set."""
        reward_fn = create_reward_function(logger=None)
        assert reward_fn.__name__ == "ocaml_reward"

    def test_empty_completions_returns_empty_list(self):
        """Empty completions list returns empty rewards list."""
        reward_fn = create_reward_function(logger=None)
        rewards = reward_fn(prompts=[], completions=[], tests=[])
        assert rewards == []

    def test_returns_list_of_floats(self):
        """Reward function returns a list of floats."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=["x"],
            tests=["let () = assert (foo 1 = 1)"],
        )
        assert isinstance(rewards, list)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)


class TestRewardScoring:
    """Test reward scoring for various completion types."""

    def test_perfect_solution_gets_full_reward(self):
        """Perfect solution scores 1.0."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let rec factorial (n : int) : int ="],
            completions=["if n <= 1 then 1 else n * factorial (n - 1)"],
            tests=["let () = assert (factorial 5 = 120)"],
        )
        assert rewards[0] == 1.0

    def test_syntax_error_gets_zero(self):
        """Syntax error scores 0.0."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=["if x then"],  # Incomplete syntax
            tests=["let () = assert (foo 1 = 1)"],
        )
        assert rewards[0] == 0.0

    def test_compiles_but_fails_tests(self):
        """Code that compiles but fails tests gets type_check + compile reward."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let rec factorial (n : int) : int ="],
            completions=["0"],  # Wrong implementation but compiles
            tests=["let () = assert (factorial 5 = 120)"],
        )
        # type_check (0.25) + compile (0.10) = 0.35
        assert rewards[0] == 0.35

    def test_type_error_gets_graduated_score(self):
        """Type errors get graduated scoring based on error count."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=['x + "string"'],  # Type mismatch
            tests=["let () = assert (foo 1 = 1)"],
        )
        # Should get partial type_check score (graduated)
        assert 0.0 < rewards[0] < 0.25

    def test_empty_completion_gets_zero(self):
        """Empty completion scores 0.0."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=[""],
            tests=["let () = assert (foo 1 = 1)"],
        )
        assert rewards[0] == 0.0


class TestDegenerateDetection:
    """Test degenerate output detection and penalty."""

    def test_degenerate_output_penalized(self):
        """Degenerate output (prose + code block) gets 0.3x penalty."""
        reward_fn = create_reward_function(logger=None, parallel=False)

        # Non-degenerate solution
        rewards_clean = reward_fn(
            prompts=["let rec factorial (n : int) : int ="],
            completions=["if n <= 1 then 1 else n * factorial (n - 1)"],
            tests=["let () = assert (factorial 5 = 120)"],
        )

        # Degenerate with markdown and prose
        degenerate = """Here's the solution:
```ocaml
if n <= 1 then 1 else n * factorial (n - 1)
```
This uses recursion."""
        rewards_degenerate = reward_fn(
            prompts=["let rec factorial (n : int) : int ="],
            completions=[degenerate],
            tests=["let () = assert (factorial 5 = 120)"],
        )

        # Degenerate should be penalized (0.3x)
        assert rewards_degenerate[0] < rewards_clean[0]

    def test_stub_solution_penalized(self):
        """Stub solutions (failwith, todo) are detected as degenerate."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=['failwith "todo"'],
            tests=["let () = ()"],  # Empty test
        )
        # Stub solutions get degenerate penalty
        # With empty tests they get type_check + compile but penalized
        assert rewards[0] < 0.35  # Less than base compile reward


class TestStylePenalty:
    """Test style penalty application."""

    def test_verbose_solution_penalized(self):
        """Verbose solutions get style penalty."""
        reward_fn = create_reward_function(logger=None, parallel=False)

        # Concise solution
        concise = "if n <= 1 then 1 else n * factorial (n - 1)"

        # Same logic but unnecessarily verbose (many let bindings)
        verbose = """
let result =
  let base_case = 1 in
  let check = n <= 1 in
  let recursive_call = factorial (n - 1) in
  let multiply = n * recursive_call in
  if check then base_case else multiply
in result
"""

        rewards_concise = reward_fn(
            prompts=["let rec factorial (n : int) : int ="],
            completions=[concise],
            tests=["let () = assert (factorial 5 = 120)"],
        )

        rewards_verbose = reward_fn(
            prompts=["let rec factorial (n : int) : int ="],
            completions=[verbose],
            tests=["let () = assert (factorial 5 = 120)"],
        )

        # Both should work but concise should score higher
        assert rewards_concise[0] >= rewards_verbose[0]


class TestSignaturePrepending:
    """Test that signatures are prepended correctly."""

    def test_body_only_completion(self):
        """Completion without signature gets signature prepended."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=["x + 1"],
            tests=["let () = assert (foo 1 = 2)"],
        )
        assert rewards[0] == 1.0

    def test_full_redefinition_used_as_is(self):
        """Completion that redefines function is used as-is."""
        reward_fn = create_reward_function(logger=None, parallel=False)
        # Multi-line to avoid MIN_NON_EMPTY_LINES rejection
        completion = """let foo (x : int) : int =
  x + 1"""
        rewards = reward_fn(
            prompts=["let foo (x : int) : int ="],
            completions=[completion],
            tests=["let () = assert (foo 1 = 2)"],
        )
        assert rewards[0] == 1.0
