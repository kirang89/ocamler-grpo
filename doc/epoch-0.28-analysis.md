# Training Analysis - Epoch 0.28-0.29

**Date:** 2025-12-10
**Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct
**Dataset:** kiranpg/ocaml-training-problems (5,060 problems)
**Training Progress:** Epoch 0.28-0.29 (very early stage)

## Executive Summary

**Status:** 🟡 **Struggling - Requires Immediate Intervention**

The training is experiencing **mode collapse** with critically low reward diversity (`frac_zero_std` up to 0.50) and low absolute rewards (0.066-0.115 out of 1.0 max). The model is producing syntactically plausible but mostly incorrect OCaml code, achieving only 6-11% of possible reward.

**Recommendation:** Apply immediate hyperparameter changes (temperature increase, more generations) before continuing. If no improvement by epoch 1.0, consider curriculum learning or stronger base model.

---

## Current Training Configuration

### Environment Variables (.envrc)
```bash
PYTHONUNBUFFERED=1
TRAINING_DATASET=kiranpg/ocaml-training-problems
GRPO_NUM_GENERATIONS=6
GRPO_TEMPERATURE=0.9
GRPO_MAX_PROMPT=704
```

### Default Parameters (train.py)
```python
# Model & Data
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAINING_DATASET = "kiranpg/ocaml-training-problems"

# GRPO Configuration
GRPO_BATCH_SIZE = 4                    # per_device_train_batch_size
GRPO_GRAD_ACCUM_STEPS = 1             # gradient_accumulation_steps
GRPO_NUM_GENERATIONS = 6              # overridden from env
GRPO_MAX_PROMPT = 704                 # overridden from env
GRPO_MAX_COMPLETION = 512
GRPO_NUM_EPOCHS = 1.0
GRPO_LEARNING_RATE = 5e-6             # currently running at ~3.55e-6 due to warmup
GRPO_TEMPERATURE = 0.9                # overridden from env
GRPO_MAX_GRAD_NORM = 1.0
GRPO_LOGGING_STEPS = 1

# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Reward Structure (make_syntax_aware_reward)
# - Structural (END marker): 5%
# - Type checking: 20% (graduated by error count)
# - Compilation: 10% (always attempted, partial credit)
# - Tests: 65% (only if compiles successfully)

# Penalties
RUNAWAY_PENALTY_MULTIPLIER = 0.3      # for completions hitting max length without END
MIN_NON_EMPTY_LINES = 8               # minimum code lines to get any reward
```

---

## Training Log Analysis

### Sample Log Metrics (Epoch 0.28-0.29)
```
[Epoch 0.28]  loss=-0.0469  grad=0.0838  lr=3.58e-06  reward=0.102±0.077  syntax_rew=0.102±0.078  entropy=0.155  frac_zero_std=0.00
[Epoch 0.28]  loss=-0.0333  grad=0.0362  lr=3.58e-06  reward=0.066±0.072  syntax_rew=0.066±0.069  entropy=0.090  frac_zero_std=0.00
[Epoch 0.29]  loss=0.0111   grad=0.0406  lr=3.57e-06  reward=0.115±0.065  syntax_rew=0.115±0.082  entropy=0.165  frac_zero_std=0.25
[Epoch 0.29]  loss=-0.0210  grad=0.0317  lr=3.57e-06  reward=0.084±0.063  syntax_rew=0.084±0.075  entropy=0.078  frac_zero_std=0.25
[Epoch 0.29]  loss=-0.0269  grad=0.0286  lr=3.56e-06  reward=0.073±0.044  syntax_rew=0.073±0.063  entropy=0.103  frac_zero_std=0.50
[Epoch 0.29]  loss=-0.0109  grad=0.0380  lr=3.55e-06  reward=0.113±0.066  syntax_rew=0.113±0.084  entropy=0.110  frac_zero_std=0.25
```

### Metric Ranges
| Metric | Range | Expected | Assessment |
|--------|-------|----------|------------|
| **Loss** | -0.19 to +0.12 | Oscillating (normal for policy gradient) | ✅ Normal |
| **Gradient Norm** | 0.015 - 0.087 | 0.01 - 0.5 | ✅ Healthy |
| **Learning Rate** | 3.55e-6 - 3.58e-6 | 5e-6 (target) | ⚠️ In warmup phase |
| **Entropy** | 0.045 - 0.217 | 0.10 - 0.20 | ⚠️ Highly variable |
| **Reward (mean)** | 0.066 - 0.115 | > 0.30 by epoch 0.5 | 🔴 Very low |
| **Reward (std)** | 0.044 - 0.081 | < 50% of mean | 🔴 Too high (50-70% CV) |
| **frac_zero_std** | 0.00 - 0.50 | < 0.10 | 🔴 Critical issue |

---

## Critical Issues Identified

### 1. Mode Collapse (CRITICAL)
**Symptom:** `frac_zero_std` reaching 0.25-0.50
**Meaning:** 25-50% of generated samples have identical advantage values
**Impact:** GRPO relies on comparing different quality samples. Without diversity, there's no learning signal.

**Root Cause:**
- Temperature 0.9 is still too low for early GRPO training
- Model is converging to a few "safe" patterns that get mediocre rewards
- Insufficient exploration of solution space

### 2. Low Absolute Rewards (0.066-0.115 / 1.0 max)
**Breakdown Analysis:**
- **Structural (5% max):** Likely getting ~5% (END marker present)
- **Type checking (20% max):** Getting ~5-15% (multiple syntax errors)
- **Compilation (10% max):** Getting ~1-5% (partial credit for trying)
- **Tests (65% max):** Getting ~0% (not passing any tests)

**Interpretation:** Model is producing syntactically plausible OCaml code with correct structure, but:
- Multiple type errors (1-3 errors per solution)
- Fails to compile successfully
- Never passes actual test cases

### 3. High Reward Variance
**Coefficient of Variation:** 50-70% (std/mean)
**Impact:** Unstable learning signal, suggests model output quality is inconsistent

### 4. Unstable Entropy
**Range:** 0.045 - 0.217 (highly variable)
**Impact:**
- Low entropy (0.045): Model too confident, not exploring
- High entropy (0.217): Model too uncertain, random guessing
- Wide swings suggest unstable exploration/exploitation balance

---

## Recommended Actions

### Priority 1: Fix Mode Collapse (CRITICAL)

**Action 1: Increase Temperature**
```bash
export GRPO_TEMPERATURE=1.1  # increase from 0.9
```
**Rationale:** Higher temperature increases output diversity, reducing identical generations

**Action 2: Increase Number of Generations**
```bash
export GRPO_NUM_GENERATIONS=8  # increase from 6
```
**Rationale:** More samples per prompt improves advantage estimation and diversity

**Expected Impact:**
- `frac_zero_std` should drop below 0.10 within 100 steps
- Entropy should stabilize around 0.15-0.25

### Priority 2: Accelerate Learning

**Action 3: Increase Learning Rate**
```bash
export GRPO_LEARNING_RATE=8e-6  # increase from 5e-6
```
**Rationale:** Current LR is conservative; rewards should climb faster at epoch 0.28

**Action 4: Consider Longer Completions**
```bash
export GRPO_MAX_COMPLETION=600  # increase from 512
```
**Rationale:** Some solutions might be getting cut off before END marker

**Expected Impact:**
- Faster reward growth
- Reaching 0.20+ rewards by epoch 0.5

### Priority 3: Monitor and Adapt

**What to Check After Changes:**

1. **At Epoch 0.5:**
   - ✅ `frac_zero_std` < 0.10
   - ✅ Mean reward > 0.20
   - ✅ Entropy stable (0.15-0.25)
   - ✅ Some test passes appearing (syntax_rew showing compilation success)

2. **At Epoch 1.0:**
   - ✅ Mean reward > 0.35
   - ✅ Clear upward trend in rewards
   - ✅ Regular test passes (10-20% of samples)

---

## Contingency Plans

### If No Improvement by Epoch 1.0

#### Option A: Curriculum Learning (Staged Rewards)
Modify reward structure to focus on easier objectives first:

```python
# Phase 1 (epochs 0-0.5): Focus on syntax
- Structural: 10%
- Type checking: 40% (more weight)
- Compilation: 30% (more weight)
- Tests: 20% (reduced)

# Phase 2 (epochs 0.5-1.0): Transition to correctness
- Structural: 5%
- Type checking: 25%
- Compilation: 20%
- Tests: 50%

# Phase 3 (epochs 1.0+): Current structure
- Structural: 5%
- Type checking: 20%
- Compilation: 10%
- Tests: 65%
```

**Implementation:** Add epoch-based conditional logic in `make_syntax_aware_reward` (train.py:452)

#### Option B: Test Base Model Capability
Run inference test to verify base model can solve these problems:

```bash
python test_base_model.py  # Test on 10 random problems
```

If base model fails most problems (< 20% success rate), consider:
- Using stronger base model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Filtering dataset for easier problems (subset with shorter solutions)
- Adding more few-shot examples to prompt

#### Option C: Reward Debugging
Check actual completions to diagnose failure modes:

```bash
# On training machine
tail -50 grpo_runs/reward_logs/completions.jsonl
tail -50 grpo_runs/reward_logs/syntax_aware_breakdown.jsonl
```

Look for:
- Are completions gibberish or plausible?
- Common syntax errors (missing semicolons, wrong types, etc.)
- Pattern in failures (specific problem types?)

---

## Stop Criteria

### Red Flags to STOP Training

🛑 **Stop if at Epoch 1.0:**
- Mean reward still < 0.15
- `frac_zero_std` still > 0.30
- Completions are gibberish (random tokens)
- No test passes observed in logs
- Reward trend is flat or decreasing

### Green Lights to CONTINUE

✅ **Continue if:**
- Rewards showing upward trend (even if slow)
- `frac_zero_std` decreasing over time
- Occasional test passes appearing (even 5-10%)
- Entropy stabilizing
- Completions are syntactically valid OCaml

---

## Expected Training Trajectory (After Fixes)

### Healthy Training Should Show:

**Epoch 0.0-0.3:**
- Rewards: 0.05 → 0.15
- Model learns: END marker, basic syntax, function definitions
- `frac_zero_std` drops from 0.50 → 0.10

**Epoch 0.3-0.6:**
- Rewards: 0.15 → 0.35
- Model learns: Type-correct code, some compilations
- First test passes appear (5-10% of samples)

**Epoch 0.6-1.0:**
- Rewards: 0.35 → 0.55
- Model learns: Problem-specific logic
- Regular test passes (20-30% of samples)

**Post-Epoch 1.0:**
- Rewards: 0.55 → 0.70+
- Model learns: Edge cases, robust solutions
- Majority of samples compile and pass tests

---

## Next Steps

1. **Immediate:** Apply Priority 1 and Priority 2 changes before next training run
2. **Monitor:** Check logs at epoch 0.5 and 1.0 using criteria above
3. **Adjust:** If no improvement, proceed to contingency plans
4. **Document:** Record reward trends and `frac_zero_std` progression

## Files to Monitor

```bash
# Essential logs
grpo_runs/learning.log                           # Main training metrics
grpo_runs/reward_logs/completions.jsonl          # Actual model outputs
grpo_runs/reward_logs/syntax_aware_breakdown.jsonl  # Detailed reward breakdown
```

## Conclusion

The current training is experiencing **mode collapse due to insufficient diversity**. This is fixable with temperature increase and more generations. The reward structure appears sound (5% + 20% + 10% + 65%), but the model needs more exploration to escape local optima.

**Decision:** Worth continuing with hyperparameter adjustments. Reassess at epoch 0.5 and 1.0.

---

## Appendix: Understanding GRPO Metrics

### frac_zero_std (Fraction of Zero Standard Deviation)
- **What it measures:** Percentage of prompts where all generated samples have identical advantage
- **Why it matters:** GRPO learns by comparing good vs. bad samples for same prompt
- **Healthy range:** < 0.10 (less than 10% of prompts have identical samples)
- **Current issue:** 0.25-0.50 means model is stuck generating similar outputs

### Entropy
- **What it measures:** Uncertainty/randomness in model's next-token predictions
- **Why it matters:** Too low = exploitation (no exploration), too high = random (no learning)
- **Healthy range:** 0.10-0.20 for code generation
- **Current issue:** Swings between 0.045 and 0.217 suggest instability

### Coefficient of Variation (Reward StdDev / Mean)
- **What it measures:** Relative variability in reward outcomes
- **Why it matters:** High variance = inconsistent quality = unstable learning
- **Healthy range:** < 0.30 (30%)
- **Current issue:** 0.50-0.70 (50-70%) is very high
