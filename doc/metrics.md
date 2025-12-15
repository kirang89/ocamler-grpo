# GRPO Learning Metrics Documentation

This document describes the essential learning metrics logged to `learning.log` during GRPO training, making it easier to monitor training progress and spot patterns.

## Overview

The `learning.log` file contains a filtered, human-readable view of the most important training metrics. Each line represents a training step with key indicators of model health and learning progress.

## Log Format

```
[Epoch X.XX]  loss=X.XXXX  grad=X.XXXX  lr=X.XXe-XX  reward=X.XXX±X.XXX  syntax_rew=X.XXX±X.XXX  entropy=X.XXX  frac_zero_std=X.XX
```

## Metrics Reference

### Core Training Metrics

#### `epoch`
- **Type**: Float (e.g., 0.60, 0.61)
- **Range**: 0.0 to num_epochs
- **Definition**: The current training epoch, representing how many times the model has seen the full dataset
- **What it means**: Training progress indicator. Each epoch represents one complete pass through the training data.
- **What to watch for**: Linear progression. Should increase steadily over time.

#### `loss`
- **Type**: Float
- **Range**: Typically -0.1 to 0.1 for GRPO (can vary)
- **Definition**: The GRPO policy loss, measuring how much the policy is being updated
- **What it means**:
  - Negative loss: Policy is being pushed toward higher reward actions
  - Positive loss: Policy is being pushed away from lower reward actions
  - Zero loss: No policy update (all samples have similar advantage)
- **What to watch for**:
  - Extremely large values (>1.0): May indicate instability
  - Persistent zeros: May indicate reward collapse or insufficient diversity
  - Oscillating sign: Normal for GRPO, indicates exploration

#### `grad_norm`
- **Type**: Float
- **Range**: Typically 0.01 to 10.0 for stable training
- **Definition**: The L2 norm of the gradients, measuring the magnitude of parameter updates
- **What it means**: How aggressively the model parameters are being updated
- **What to watch for**:
  - Values > 10: Potential gradient explosion, may need gradient clipping
  - Values near 0: Minimal learning, may indicate reward saturation
  - Sudden spikes: Can indicate batch instability or edge cases

#### `learning_rate`
- **Type**: Float (scientific notation)
- **Range**: Typically 1e-6 to 1e-5 for fine-tuning
- **Definition**: The current learning rate (may decay over training)
- **What it means**: Step size for parameter updates. Smaller = more conservative updates.
- **What to watch for**: Should follow your scheduler (e.g., linear decay, cosine)

### Reward Metrics

#### `reward`
- **Type**: Float
- **Range**: 0.0 to 1.0 (for this codebase)
- **Definition**: Mean reward across all completions in the batch
- **What it means**: Average quality of generated solutions. Combines structural, type-checking, compilation, and test success.
- **What to watch for**:
  - Increasing trend: Model is improving
  - Plateau at < 0.3: Model stuck in low-quality local optimum
  - Sudden drops: Possible reward collapse or distribution shift
  - Values > 0.8: Strong performance (if tests are challenging)

#### `reward_std`
- **Type**: Float
- **Range**: 0.0 to ~0.2
- **Definition**: Standard deviation of rewards across batch completions
- **What it means**: Diversity of solution quality within a batch
- **What to watch for**:
  - Near zero: Model producing very similar quality outputs (may indicate mode collapse)
  - Increasing: Model exploring diverse strategies (good early in training)
  - Decreasing over time: Model converging to consistent quality (good late in training)

#### `rewards/syntax_aware_reward/mean`
- **Type**: Float
- **Range**: 0.0 to 1.0
- **Definition**: Mean reward from the syntax-aware reward function specifically
- **What it means**:
  - 0.00-0.05: Structural credit only (has END marker)
  - 0.05-0.35: Type-checking partial credit (has type errors)
  - 0.35-0.45: Type-checks and compiles but tests fail
  - 0.45-1.00: Tests pass (0.65 + structural/type bonuses)
- **What to watch for**: Should correlate with overall reward. Measures OCaml-specific quality.

#### `rewards/syntax_aware_reward/std`
- **Type**: Float
- **Range**: 0.0 to ~0.2
- **Definition**: Standard deviation of syntax-aware rewards
- **What it means**: Diversity in OCaml code quality within batch
- **What to watch for**: Similar to reward_std. Low values may indicate consistency or mode collapse.

### Batch Pass Metrics
(Logged to `batch_metrics.jsonl`)

#### `pass_at_1`
- **Type**: Float
- **Range**: 0.0 to 1.0
- **Definition**: The average percentage of correct solutions per problem in a batch.
- **How it's calculated**:
  1. For each problem in the batch, calculate the fraction of generated solutions that passed all tests.
     - Example: Problem A has 1/4 correct, Problem B has 2/4 correct.
  2. Average these fractions across all problems in the batch.
     - Example: `(0.25 + 0.50) / 2 = 0.375`.
- **What it means**: The probability that a single generated solution is correct.
- **What to watch for**: Steady increase indicates the model is getting better at solving problems on the first try.

#### `pass_at_all`
- **Type**: Float
- **Range**: 0.0 to 1.0
- **Definition**: The percentage of problems in a batch that had **at least one** correct solution.
- **How it's calculated**:
  1. For each problem, check if *any* of its generated solutions passed all tests.
  2. Calculate the percentage of problems where this is true.
     - Example: If Problem A has 1/4 correct (solved) and Problem B has 0/4 correct (unsolved), Pass@All is `0.5` (50%).
- **What it means**: The model's ability to eventually solve a problem given multiple attempts (`num_generations`).
- **What to watch for**: This should be higher than Pass@1. If it plateaus while Pass@1 rises, the model might be overfitting to easy problems while failing hard ones.

### Policy Health Metrics

#### `entropy`
- **Type**: Float
- **Range**: 0.0 to ~0.15 (typical for trained models)
- **Definition**: Average entropy of the policy's token probability distribution
- **What it means**:
  - High entropy (>0.1): Model is uncertain, exploring diverse outputs
  - Low entropy (<0.05): Model is confident, generating deterministic outputs
- **What to watch for**:
  - Decreasing over time: Normal (model becoming more confident)
  - Near zero: May indicate mode collapse or overfitting
  - Increasing: Model becoming less certain (may indicate instability)
  - Sudden drops to ~0.04: Possible mode collapse to template solutions

#### `frac_reward_zero_std`
- **Type**: Float
- **Range**: 0.0 to 1.0
- **Definition**: Fraction of problems in the batch where all completions got identical rewards
- **What it means**:
  - 0.0: Every problem has diverse solution quality
  - 0.5: Half the problems have all completions at same reward
  - 1.0: All problems have identical reward across completions (mode collapse)
- **What to watch for**:
  - Values > 0.8: Strong sign of mode collapse
  - Values > 0.5: Model may be converging too strongly
  - Values near 0: Healthy exploration

## Interpreting Common Patterns

### Healthy Training
```
[Epoch 0.10]  loss=-0.0311  grad=0.028  lr=2.00e-06  reward=0.207±0.006  syntax_rew=0.207±0.012  entropy=0.055  frac_zero_std=0.75
[Epoch 0.20]  loss=-0.0415  grad=0.035  lr=1.95e-06  reward=0.315±0.015  syntax_rew=0.315±0.018  entropy=0.068  frac_zero_std=0.50
[Epoch 0.30]  loss=-0.0520  grad=0.042  lr=1.90e-06  reward=0.428±0.022  syntax_rew=0.428±0.025  entropy=0.074  frac_zero_std=0.25
```
- Reward increasing steadily
- Reward diversity (std) increasing (exploration)
- Entropy increasing (model exploring)
- frac_zero_std decreasing (less mode collapse)

### Mode Collapse Warning
```
[Epoch 0.50]  loss=0.0000  grad=0.000  lr=1.85e-06  reward=0.210±0.000  syntax_rew=0.210±0.000  entropy=0.044  frac_zero_std=1.00
[Epoch 0.51]  loss=0.0000  grad=0.000  lr=1.84e-06  reward=0.210±0.000  syntax_rew=0.210±0.000  entropy=0.044  frac_zero_std=1.00
```
- Zero loss and grad (no learning)
- Zero reward std (all solutions identical)
- Low entropy (deterministic outputs)
- frac_zero_std = 1.0 (complete collapse)

### Runaway Completions
```
[Epoch 0.25]  loss=-0.0311  grad=0.028  lr=1.98e-06  reward=0.050±0.000  syntax_rew=0.050±0.000  entropy=0.055  frac_zero_std=1.00
```
- Reward capped at 0.05 (RUNAWAY_REWARD_CAP)
- Indicates completions hit max length without END marker
- Model generating extremely long outputs (filibustering)

### Gradient Instability
```
[Epoch 0.35]  loss=0.0947  grad=15.786  lr=1.90e-06  reward=0.207±0.012  entropy=0.103  frac_zero_std=0.75
[Epoch 0.36]  loss=-0.0312  grad=0.024  lr=1.89e-06  reward=0.207±0.006  entropy=0.080  frac_zero_std=0.75
```
- Sudden grad_norm spike (15.786)
- May need gradient clipping or lower learning rate
- Could be triggered by edge case in batch

## How to Use This Log

1. **Monitor during training**: `tail -f grpo_runs/learning.log`
2. **Quick health check**: Look for increasing reward, reasonable entropy (0.05-0.10)
3. **Detect mode collapse**: Watch for frac_zero_std > 0.8 or entropy < 0.04
4. **Spot instability**: Look for grad_norm > 10 or oscillating loss
5. **Track progress**: Plot reward over epochs to see learning curve

## Related Files

- Full training logs with all metrics: Check trainer output or WandB
- Reward breakdown details: `grpo_runs/reward_logs/syntax_aware_breakdown.jsonl`
- Generated completions: `grpo_runs/reward_logs/completions.jsonl`
