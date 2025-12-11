# Gradient Explosion Issue and Fix

**Date:** 2025-12-10
**Status:** Fixed
**Related Files:** `train.py`, `doc/runaway-penalty-plan.md`

## Problem Summary

Training experienced catastrophic gradient explosion at Epoch 0.18:
- **Normal gradient:** 0.1-3.0
- **Warning sign:** 111.17 (Epoch 0.17)
- **Explosion:** 54,648,652 (Epoch 0.18)
- **Result:** Training collapse, model weights corrupted

## Root Causes

### 1. Missing Gradient Clipping
The `GRPOConfig` had no `max_grad_norm` parameter, allowing unbounded gradient magnitudes.

### 2. GRPO Importance Sampling Mechanics
GRPO computes gradients using importance ratios that can explode:

```
gradient ∝ (π_new / π_ref) × advantage
```

Where:
- `π_new` = probability under current policy
- `π_ref` = probability under reference policy
- `advantage` = relative quality of this sample

### 3. Multiplicative Penalty Amplified Risk
The recent switch from hard reward cap (0.05) to multiplicative penalty (0.3x) restored reward diversity:
- **Good:** Enabled learning (eliminated zero-gradient collapse)
- **Bad:** Created large advantages that amplified policy ratio explosions

## Detailed Mechanism

### Example: How Gradients Explode

**Scenario:** Generating OCaml function, choosing token `fold_left`

**Early Training (Reference Policy):**
```
π_ref(fold_left) = 0.00001  (0.001% - very uncertain)
```

**Later Training (New Policy):**
```
π_new(fold_left) = 0.9999   (99.99% - very confident)
```

**Importance Ratio:**
```
ratio = π_new / π_ref
      = 0.9999 / 0.00001
      = 99,990x
```

**If this completion has high advantage (+0.80):**
```
gradient_per_token = 99,990 × 0.80 = 79,992

Across 200 tokens:
gradient_total = 79,992 × 200 = 15,998,400

→ Training explodes
```

### Why It Happened in This Training

1. Model learned good solutions (rewards climbed from 0.077 → 0.203)
2. Policy shifted strongly toward high-reward behaviors
3. For some tokens, `π_new / π_ref` became extremely large (10,000x+)
4. Large advantages (from multiplicative penalty) amplified the effect
5. Without gradient clipping, one bad sequence destroyed the entire update

## Timeline of Events

### Before Fix: Hard Cap Era
```
Rewards: 0.044-0.050 (capped)
Gradients: 0.0000 (no diversity)
frac_zero_std: 0.50-1.00
Status: No learning (gradient collapse)
```

### Transition: Multiplicative Penalty
```
Rewards: 0.077-0.203 (climbing)
Gradients: 0.1-3.0 (healthy learning)
frac_zero_std: 0.00-0.75 (much better)
Status: Learning! But unstable
```

### Explosion Point
```
[Epoch 0.17]  grad=111.1711    (warning)
[Epoch 0.18]  grad=54648652.0000  (catastrophic)
Status: Training destroyed
```

## Solution Implemented

### Fix 1: Add Gradient Clipping

**File:** `train.py:654-670`

```python
# Gradient clipping to prevent training instability
max_grad_norm = float(os.environ.get("GRPO_MAX_GRAD_NORM", "1.0"))

return GRPOConfig(
    # ... other params ...
    max_grad_norm=max_grad_norm,  # Clip gradients to prevent explosions
    # ... other params ...
)
```

**How it works:**
```python
if gradient_norm > max_grad_norm:
    gradient = gradient × (max_grad_norm / gradient_norm)
```

**Effect:**
- Gradient of 54,648,652 → clipped to 1.0
- Preserves direction, limits magnitude
- Prevents catastrophic updates

### Fix 2: Temperature Adjustment (Recommended)

**Purpose:** Increase generation diversity to reduce identical samples

**Current:** `GRPO_TEMPERATURE=0.7`
**Recommended:** `GRPO_TEMPERATURE=0.8`

**Benefits:**
- Reduces `frac_zero_std` (fewer identical samples)
- More reward variance → better advantage estimates
- Prevents mode collapse

## Configuration for Stable Training

### Recommended Settings

```bash
# Core stability
export GRPO_MAX_GRAD_NORM=1.0           # Prevent gradient explosion
export RUNAWAY_PENALTY_MULTIPLIER=0.3   # Preserve reward diversity
export GRPO_TEMPERATURE=0.8             # Increase sample diversity

# Optional: Extra safety
export GRPO_LEARNING_RATE=3e-6          # Down from 5e-6 (more conservative)
```

### Conservative Settings (If Issues Persist)

```bash
export GRPO_MAX_GRAD_NORM=0.5           # Tighter clipping
export GRPO_LEARNING_RATE=3e-6          # Lower LR
export RUNAWAY_PENALTY_MULTIPLIER=0.4   # Less aggressive penalty
export GRPO_TEMPERATURE=0.9             # Maximum diversity
```

## Monitoring Guidelines

### Healthy Training ✓

```
[Epoch X]  loss=0.05-0.5   grad=0.05-0.99  reward=0.10-0.30  frac_zero_std=0.0-0.5
```

**Indicators:**
- Gradients stay below 1.0 (clipping working)
- Rewards gradually increase
- `frac_zero_std` mostly low
- Entropy stable (0.15-0.25)

### Warning Signs ⚠️

```
[Epoch X]  grad=0.99-1.00  (consistently at clip limit)
[Epoch X]  frac_zero_std=0.75-1.00  (frequently)
```

**Actions:**
- If gradients always at 1.0: raise `max_grad_norm` to 1.5-2.0
- If high `frac_zero_std`: increase temperature to 0.9

### Danger Signs 🚨

```
[Epoch X]  loss=0.0000  grad=0.0000  (reward collapse)
[Epoch X]  reward decreasing over time
```

**Actions:**
- Reward collapse: adjust `RUNAWAY_PENALTY_MULTIPLIER` (increase to 0.4-0.5)
- Decreasing rewards: lower temperature, reduce LR

## Key Insights

### 1. Gradient Clipping is Not Optional
- **Required** for all policy gradient methods (GRPO, PPO, REINFORCE)
- Should have been included from the start
- Industry standard practice

### 2. Reward Diversity vs Stability Trade-off
- Hard cap (0.05): Stable but no learning
- No cap: Learning but explosive
- Multiplicative penalty + clipping: **Best of both worlds**

### 3. RL Training is Inherently Unstable
Unlike supervised learning, RL involves:
- Policy ratios (can explode exponentially)
- Advantages (can be noisy/extreme)
- Non-stationary distributions (policy keeps changing)
- Long sequences (errors compound)

**Multiple safeguards needed:**
- ✅ Gradient clipping
- ✅ Reward shaping (multiplicative penalty)
- ✅ Temperature tuning
- ✅ Conservative learning rate

## Related Issues

### Previously Solved: Zero Gradient Collapse
- **Problem:** Hard reward cap (0.05) caused all samples to get identical rewards
- **Solution:** Multiplicative penalty (0.3x) - see `doc/runaway-penalty-plan.md`
- **Status:** Fixed

### Previously Solved: Filibuster Training
- **Problem:** Model hitting max length without END marker
- **Solution:** Runaway penalty - see `doc/fix-filibuster-training.md`
- **Status:** Fixed

## Next Steps

1. **Start fresh training** with gradient clipping enabled
2. **Monitor logs** for first 0.1 epochs - ensure gradients stay <1.0
3. **Adjust temperature** if `frac_zero_std` frequently >0.5
4. **Track reward progression** - should increase steadily to 0.3-0.5+
5. **Save checkpoints** every 100 steps for safety

## Summary

**Root cause:** Missing gradient clipping + importance ratio explosion
**Fix:** Added `max_grad_norm=1.0` to GRPOConfig
**Status:** Ready for stable training
**Confidence:** High - this is standard RL practice

The combination of gradient clipping + multiplicative penalty + temperature tuning provides a robust foundation for stable GRPO training.
