 Key Findings on End Markers in RLVR/GRPO

  1. Standard Practice: Use Native EOS Tokens, Not Custom Markers

  Modern GRPO implementations rely on the model's native EOS (End of Sequence) token like <|endoftext|>
   rather than custom markers like (* END *). The training handles this by:

  - Creating binary masks for generated tokens so that tokens after the first EOS are ignored
  - Monitoring clipped_ratio to track completions that didn't end with EOS before hitting max_length
  (indicates truncation issues)
  - Tracking mean_terminated_length for completions that successfully ended with EOS

  A high clipped_ratio signals problems - the model is hitting max_length instead of naturally
  terminating.

  Source: https://rlhfbook.com/c/11-policy-gradients.html

  2. RLVR Uses Binary Rewards (0/1) Based on Execution

  In RLVR for code generation:
  - Code interpreter executes the generated code → 0/1 reward for fail/pass
  - Battery of unit tests can be run → binary reward
  - No intermediate structural rewards for markers or formatting

  This is much simpler than your current multi-stage reward (structural 5% + type 20% + compile 10% +
  tests 65%).

  Sources:
  - https://www.theainavigator.com/blog/what-is-reinforcement-learning-with-verifiable-rewards-rlvr
  - https://labelbox.com/solutions/reinforcement-learning-with-verifiable-rewards/

  3. Custom Markers Create Known Problems

  Research shows that structural/format tokens can be unintentionally penalized during training,
  degrading formatting and stability over time. The issues include:

  - Models getting stuck in "redundant verification loops"
  - Failing to produce final answers within token limits
  - Format tokens being penalized, leading to degraded structure

  Source: https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward

  4. Reasoning Models Use Structured Tags Differently

  For reasoning tasks (not pure code generation), GRPO uses tags like:
  - <think>reasoning</think> <answer>solution</answer>
  - {reasoning_start}...{reasoning_end} {solution_start}...{solution_end}

  But these get format rewards (e.g., 3 points for perfect structure, 0.5 per correct tag) and are part
   of the reasoning process, not stopping criteria.

  Source: https://youssefh.substack.com/p/gemma-3-reasoning-fine-tuning-with-4a0

  5. Reward Assignment Strategy Differs from PPO

  Critically: GRPO assigns the final reward to EVERY token in the sequence (uniform), while PPO with a
  value network assigns different values per token, discounting from the EOS token.

  This means intermediate structural rewards like "+0.05 for END marker" don't align well with how GRPO
   propagates gradients.

  Source: https://rlhfbook.com/c/11-policy-gradients.html

  6. Best Practice for Length Control

  Instead of custom runaway penalties:
  - Set max_completion_length (you have 600)
  - Only run reward model on sequences that hit EOS token
  - Assign penalty for generating too long (exceeded max length)
  - Use "overlong episode filtering" to skip unfinished episodes

  Source: https://rlhfbook.com/c/11-policy-gradients.html

  ---
  🎯 What This Means for Your Approach

  ✅ Your Instinct is Correct

  Removing (* END *) aligns with standard practice:
  1. Research confirms: Custom markers cause training instability
  2. Standard approach: Use native EOS token
  3. Simpler is better: Binary rewards (pass/fail) are the RLVR standard

  ✅ Simplified Reward Design

  The RLVR approach would be:

  def compute_reward(completion, tests):
      # Binary: Does it work?
      if not compiles(completion):
          return 0.0

      tests_passed = run_tests(completion)
      total_tests = len(tests)

      # Simple graduated test reward
      return tests_passed / total_tests

  Plus penalties for degenerate patterns:
  if has_prose(completion):     reward *= 0.1
  if is_repetitive(completion): reward *= 0.3

  ⚠️ But Consider

  Your current multi-stage reward (type checking 20%, compile 10%) provides partial credit for
  progress. Pure binary RLVR might make the sparse reward problem worse (98% → 100% zeros).

  Hybrid approach might work best:
  - Remove END marker (use native EOS)
  - Keep graduated type/compile rewards for partial credit
  - Simplify to: type_score + compile_score + test_score
  - Add prose/repetition penalties
  - Add entropy bonus

  ---
  Recommendation

  Do: Remove (* END *) marker, rely on native EOS token
  Do: Add prose penalty (aligns with RLVR principles)
  Do: Simplify reward structure
  Don't: Go fully binary (0/1) - you need partial credit given OCaml's difficulty
  Must: Still add entropy bonus to address 98% failure problem

  Your simplified design + entropy bonus is the sweet spot.

  ---
  Sources

  - https://rlhfbook.com/c/11-policy-gradients.html
  - https://www.theainavigator.com/blog/what-is-reinforcement-learning-with-verifiable-rewards-rlvr
  - https://labelbox.com/solutions/reinforcement-learning-with-verifiable-rewards/
  - https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward
  - https://youssefh.substack.com/p/gemma-3-reasoning-fine-tuning-with-4a0
  - https://huggingface.co/docs/trl/en/grpo_trainer


--------

  Question 1: How to Modify train.py for Native EOS

  Here are the specific changes needed:

  Change 1: Remove END marker from prompt (line 59)

  # Current (line 54-59)
  PROMPT_TEMPLATE = textwrap.dedent(
      """
      You are an expert OCaml engineer. Read the programming problem below and implement the solution.
      The problem specifies the function signature - you must use exactly that function name as your entry
  point.
      Provide only the implementation code without any test cases. Keep your solution concise (under ~200
  lines)
      and end the response with the exact marker `(* END *)`. Do not emit any prose, explanations, or trailing
  text after the marker.
      ...

  Remove the sentence about END marker:
  # Proposed
  PROMPT_TEMPLATE = textwrap.dedent(
      """
      You are an expert OCaml engineer. Read the programming problem below and implement the solution.
      The problem specifies the function signature - you must use exactly that function name as your entry
  point.
      Provide only the implementation code without any test cases. Keep your solution concise (under ~200
  lines).
      ...

  Change 2: Remove END_MARKER constant and scoring (lines 104, 176-177)

  # Remove these lines
  END_MARKER = "(* END *)"  # Line 104

  def score_has_end_marker(completion: str) -> float:  # Lines 176-177
      return 1.0 if completion.strip().endswith(END_MARKER) else 0.0

  Change 3: Remove structural reward for END marker (lines 524-527)

  # Current (lines 524-527)
  structural_score = 0.0
  if completion.strip().endswith(END_MARKER):
      structural_score += 0.05

  # Proposed - remove entirely since structural_score would always be 0.0
  # Just delete these lines

  Change 4: Replace runaway penalty with length-based penalty (lines 618-626)

  # Current (lines 618-626)
  is_runaway = len(completion) >= 500 and not completion.strip().endswith(END_MARKER)
  if is_runaway:
      total_reward = 0.0
      penalty_applied = True
  else:
      penalty_applied = False

  # Proposed - penalize very long completions (likely truncated or filibustering)
  # Option A: Simple length penalty
  is_excessive = len(completion) >= 550  # Near max_completion_length (600)
  if is_excessive:
      total_reward *= 0.3  # Or use RUNAWAY_PENALTY_MULTIPLIER
      penalty_applied = True
  else:
      penalty_applied = False

  # Option B: Token-based (more accurate but requires tokenization)
  # Check if completion hit max_completion_length by tokenizing
  # tokens = tokenizer.encode(completion)
  # if len(tokens) >= config.max_completion_length - 5:  # Within 5 tokens of max
  #     total_reward *= 0.3

  Simplest approach: Remove runaway penalty entirely. If the model generates garbage until max_length, it will
  fail compilation/tests anyway (0 reward).

  ---
  Question 2: Do Binary Rewards Actually Work? The Evidence is Surprising!

  🚨 Critical Issue Discovered

  A detailed analysis reveals a fundamental problem with DeepSeek's claimed binary reward approach:

  "Where does the relative advantage come from when all incorrect answers receive the same reward?"

  GRPO updates models based on relative ranking of responses. If every wrong answer gets 0 reward, there's no
  meaningful way to compare them — each incorrect response is equally bad. How does the model learn which
  incorrect responses are "better" if the reward provides no differentiation?

  The hypothesis: DeepSeek likely uses hidden partial credit mechanisms not disclosed in their paper, or their
  "rule-based" rewards aren't as purely binary as claimed.

  Source:
  https://medium.com/intelligence-factory/deepseeks-lies-a-closer-look-at-grpo-implementation-dea4607842e9

  ✅ What Actually Works in Practice

  Research shows successful GRPO code generation uses multiple reward signals, not pure binary:

  1. DeepSeek's Actual Approach (Not Pure Binary!)

  DeepSeek uses two types of rewards:
  - Accuracy Rewards: Binary (0/1) for correct final answer
  - Format Rewards: Graduated rewards for structured output (e.g., <think>...</think> tags)

  Source: https://huggingface.co/blog/NormalUhr/deepseek-r1-explained

  2. Pass@k with Advantage Shaping

  Pass@k training provides differentiation even with binary correctness rewards by:
  - Sampling k solutions per problem
  - Using advantage shaping to create surrogate rewards that differentiate failures
  - High k encourages exploration, low k encourages exploitation
  - Annealing k during training balances both

  This transforms binary rewards into meaningful gradients by looking at relative performance within groups.

  Source: https://arxiv.org/html/2510.23049

  3. Entropy-Based Reward Shaping (GTPO/GRPO-S)

  Adding entropy bonuses to rewards:
  - Achieves superior pass@k performance (exploration metric)
  - Maintains stable entropy during training
  - Outperforms baseline GRPO consistently

  Source: https://arxiv.org/html/2508.04349

  4. Rubric-Based Graduated Rewards (RGR-GRPO)

  Uses question-specific rubrics with:
  - Factual criteria: Partial credit for correct intermediate steps
  - Process criteria: Rewards for valid reasoning approaches
  - Adaptive weights for each criterion
  - Shows superior pass@k performance vs binary rewards

  Source: https://arxiv.org/html/2511.12344

  ---
  🎯 Direct Answer to Your Questions

  Question 2: Does Simple Binary (0/1) Actually Work?

  No, not in isolation. The evidence shows:

  | Approach                             | Works?    | Evidence
     |
  |--------------------------------------|-----------|---------------------------------------------------------
  ---|
  | Pure binary (0/1 for pass/fail only) | ❌ No     | Can't differentiate between failures, GRPO needs
  gradients |
  | Binary + Format rewards              | ✅ Yes    | DeepSeek's actual approach
     |
  | Binary + Advantage shaping (Pass@k)  | ✅ Yes    | Creates surrogate rewards from groups
     |
  | Binary + Entropy bonus               | ✅ Yes    | GTPO/GRPO-S outperforms baseline
     |
  | Graduated/Rubric rewards             | ✅✅ Best | RGR-GRPO shows superior performance
     |

  Critical Insight for Your Situation

  Your 98% failure rate makes binary rewards especially problematic:
  - With binary: 98% of samples get 0, 2% get 1 → almost no gradient
  - With graduated: Failures get 0.02-0.35 based on progress → clear gradient

  Your current graduated approach (type 20%, compile 10%, tests 65%) is actually closer to what works in
  practice!

  The research validates keeping graduated rewards, but suggests:

  1. ✅ Remove END marker (eliminate gaming vector)
  2. ✅ Add prose penalty (prevent instruction-tuning artifacts)
  3. ✅ Add entropy bonus (explicit exploration incentive)
  4. ✅ Keep graduated type/compile rewards (provide learning signal for 98% failures)
  5. ✅ Consider pass@k training (generate multiple solutions per problem, rank relatively)

  ---
  Recommended Implementation

  Combine the best of both approaches:

  # Simplified reward structure (no END marker)
  def make_syntax_aware_reward(evaluator, logger):
      def reward_func(...):
          # Core graduated rewards (learning signal for failures)
          type_score = graduated_type_checking(completion)     # 0-20%
          compile_score = graduated_compile(completion)        # 0-10%
          test_score = 70% * (tests_passed / total_tests)      # 0-70% (graduated!)

          base_reward = type_score + compile_score + test_score

          # Penalties for degenerate patterns (like DeepSeek's format rewards)
          if has_prose_patterns(completion):
              base_reward *= 0.1        # 90% penalty

          if is_repetitive(completion):
              base_reward *= 0.3        # 70% penalty

          # Entropy bonus (like GTPO/GRPO-S)
          entropy_bonus = 0.02  # Tune 0.01-0.05
          # Note: entropy would come from GRPO logs, not computed here

          return base_reward  # Entropy added by GRPO framework

      return reward_func

  Why this works:
  - Graduated rewards provide signal when 98% fail (unlike binary)
  - Penalties prevent gaming (like DeepSeek's format rewards)
  - Entropy bonus encourages exploration (like GTPO/GRPO-S)
  - No artificial END marker constraint

  ---
  Sources

  - https://medium.com/intelligence-factory/deepseeks-lies-a-closer-look-at-grpo-implementation-dea4607842e9
  - https://huggingface.co/blog/NormalUhr/deepseek-r1-explained
  - https://arxiv.org/html/2510.23049
  - https://arxiv.org/html/2508.04349
  - https://arxiv.org/html/2511.12344
  - https://arxiv.org/html/2508.10751v1
