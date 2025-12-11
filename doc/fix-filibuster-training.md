# Fix Filibuster Mode: Emergency Training Intervention

## Problem
Model is stuck in "filibuster mode":
- **100%** of completions hit max token limit (512 tokens)
- Missing `(* END *)` marker on all generations
- Zero gradient (all generations identical → same reward → zero advantage)
- Reward stuck at ~0.21
- Entropy collapsed to ~0.05
- **No learning happening**

## Root Cause
The model has learned a degenerate policy: generate valid-ish code indefinitely until forcibly cut off. This yields consistent partial rewards (~0.21) but prevents the model from learning to complete properly.

---

## Solution: Three Complementary Fixes

These fixes work together to address the problem at multiple levels:

1. **Stop Sequence** - Prevent filibustering at generation time
2. **Reward Cap** - Penalize filibustering if it happens anyway
3. **Increase Generations** - Add diversity to escape local minimum

All three are small, low-risk changes that target the same root cause.

---

## Implementation

### Fix 1: Add Stop Sequence (CRITICAL)

**File:** `train.py` in `create_grpo_config()` function
**Location:** Around line 548, inside the `GRPOConfig(...)` initialization

**Add this parameter:**
```python
def create_grpo_config(temperature=None) -> GRPOConfig:
    # ... existing code ...

    return GRPOConfig(
        temperature=temperature,
        top_p=0.95,
        stop_strings=["(* END *)"],  # NEW: Force stop at END marker
        output_dir=GRPO_OUTPUT_DIR,
        per_device_train_batch_size=per_device_batch,
        # ... rest of config ...
    )
```

**Why:**
- Forces generation to stop when `(* END *)` is produced
- Prevents physical filibustering at the tokenizer level
- Most direct fix to the problem
- Zero performance cost

---

### Fix 2: Add Aggressive Reward Cap for Runaways

**File:** `train.py` in `make_syntax_aware_reward()` function

#### Step 2a: Add Configuration Constant
**Location:** Top of file, near other constants (around line 71)

```python
# Reward cap for completions that hit max length without END marker
# Caps reward at 0.05 (only minimal type-check credit) to create strong
# gradient against filibustering while preserving tiny quality ranking
RUNAWAY_REWARD_CAP = float(os.environ.get("RUNAWAY_REWARD_CAP", "0.05"))
```

#### Step 2b: Apply Reward Cap
**Location:** After computing `total_reward` (around line 473), before appending to rewards list

```python
# === Final Reward ===
total_reward = structural_score + type_score + compile_score + test_score

# Apply aggressive cap for runaway completions
# (hit max length without END marker)
is_runaway = len(completion) >= 500 and not completion.strip().endswith(END_MARKER)
if is_runaway:
    # Cap at minimal reward - keeps tiny type-check credit only
    # This creates a cliff while preserving minimal quality signal
    total_reward = min(total_reward, RUNAWAY_REWARD_CAP)
    penalty_applied = True
else:
    penalty_applied = False

rewards.append(total_reward)
```

**Why:**
- Creates strong negative signal (0.21 → 0.05 is dramatic drop)
- Preserves minimal ranking between runaways (0.21→0.05, 0.18→0.045, etc.)
- Avoids pure zeroing which guarantees zero gradient if all 4 are runaways
- Configurable via env var for tuning

#### Step 2c: Update Detailed Logging
**Location:** Update the `detailed_logs.append()` call (around line 477)

```python
detailed_logs.append(
    {
        "problem_id": pid,
        "total_reward": float(total_reward),
        "structural": float(structural_score),
        "type_check": float(type_score),
        "compile": float(compile_score),
        "tests": float(test_score),
        "syntax_errors": syntax_errors if "syntax_errors" in locals() else None,
        "error_sample": error_details if "error_details" in locals() else None,
        "runaway_capped": penalty_applied,  # NEW
        "preview": completion[:200],
    }
)
```

#### Step 2d: Update Completion Logging
**Location:** Update the `completion_logs.append()` call (around line 491)

```python
completion_logs.append(
    {
        "problem_id": pid,
        "reward": float(total_reward),
        "length": len(completion),
        "runaway_capped": penalty_applied,  # NEW
        "completion": completion,
    }
)
```

---

### Fix 3: Increase Number of Generations

**File:** `train.py` in `create_grpo_config()` function
**Location:** Around line 531

**Change default from 4 to 6:**
```python
# OLD:
num_generations = int(os.environ.get("GRPO_NUM_GENERATIONS", "4"))

# NEW:
num_generations = int(os.environ.get("GRPO_NUM_GENERATIONS", "6"))
```

**Why:**
- More generations → more diversity in sampling
- Better chance of finding non-filibuster solutions
- Helps escape local minimum
- Adjust to 8 if VRAM allows, or back to 4 if OOM

**VRAM Note:** Monitor GPU memory. If you hit OOM:
- Reduce back to 4, or
- Reduce `per_device_batch_size` from 4 to 3, or
- Reduce `max_completion_length` from 512 to 384

---

## Testing & Monitoring

### Before Training
```bash
# Verify code loads without errors
python train.py --help

# Optional: Adjust reward cap if needed
export RUNAWAY_REWARD_CAP=0.03  # More aggressive
export RUNAWAY_REWARD_CAP=0.08  # More lenient
```

### During Training (First 100 Steps)

Monitor these metrics in training logs:

1. **Clipped Ratio** - Should drop below 1.0
   ```
   completions/clipped_ratio: 1.0  →  should decrease to 0.8, 0.5, etc.
   ```

2. **Gradient Norm** - Should become non-zero
   ```
   grad_norm: 0.0  →  should show values like 0.05, 0.1, etc.
   ```

3. **Reward** - Should start climbing
   ```
   reward/mean: 0.21  →  should increase toward 0.3, 0.4, etc.
   ```

4. **Entropy** - Should increase
   ```
   entropy: 0.05  →  should rise to 0.1, 0.2, etc.
   ```

### Check Logs
```bash
# Inspect completions to see if END marker appears
tail -f grpo_runs/reward_logs/completions.jsonl | jq '.completion' | grep 'END'

# Check how often runaway cap is applied
grep '"runaway_capped": true' grpo_runs/reward_logs/syntax_aware_breakdown.jsonl | wc -l
```

---

## Expected Outcomes

### Immediate (First 100 Steps)
- Stop sequence prevents most filibusters at generation time
- Clipped ratio drops from 1.0 to <0.8
- Some completions now include `(* END *)` marker
- Non-zero gradients appear

### Short-term (Steps 100-500)
- Reward climbs from 0.21 toward 0.3+
- Model learns to finish completions properly
- Entropy increases (more diverse generations)
- Structural score (0.05) starts being awarded

### Long-term (Steps 500+)
- Model generates complete, well-formed solutions
- Test success rate increases
- Reward approaches theoretical max (~1.0)

---

## Rollback / Tuning

### If Reward Drops Catastrophically
The cap might be too aggressive:
```bash
export RUNAWAY_REWARD_CAP=0.10  # Less aggressive cliff
```

### If Still 100% Clipped After 100 Steps
1. **Check stop sequence**: Verify it's actually being used in generation
2. **Increase cap aggressiveness**:
   ```bash
   export RUNAWAY_REWARD_CAP=0.02  # Even more aggressive
   ```
3. **Increase generations**:
   ```bash
   export GRPO_NUM_GENERATIONS=8
   ```

### If OOM (Out of Memory)
```bash
# Option 1: Reduce generations
export GRPO_NUM_GENERATIONS=4

# Option 2: Reduce batch size
export GRPO_BATCH_SIZE=3

# Option 3: Reduce max completion length
export GRPO_MAX_COMPLETION=384
```

### Complete Rollback
Revert all changes and set:
```bash
export RUNAWAY_REWARD_CAP=1.0  # Disables capping (no penalty)
export GRPO_NUM_GENERATIONS=4   # Original value
# Remove stop_strings from GRPOConfig
```

---

## Summary of Changes

| Component | Lines Changed | Risk Level | Impact |
|-----------|---------------|------------|--------|
| Stop sequence | 1 line | Very Low | High - prevents filibustering |
| Reward cap constant | 5 lines | Very Low | Medium - configurable penalty |
| Reward cap logic | 8 lines | Low | High - creates gradient |
| Logging updates | 2 fields × 2 places | Very Low | Medium - tracking |
| Increase generations | 1 line | Low | Medium - more diversity |
| **Total** | **~20 lines** | **Low** | **High** |

All changes are:
- ✅ Simple and easy to understand
- ✅ Configurable via environment variables
- ✅ Non-breaking (can be disabled)
- ✅ Well-documented with inline comments
- ✅ Target the same root cause from multiple angles

---

## Next Steps After This Fix

Once training is unstuck and showing progress:

1. **Monitor for overfitting** - Watch validation metrics
2. **Tune hyperparameters** - Learning rate, temperature, etc.
3. **Increase dataset size** - If model plateaus
4. **Consider longer context** - If solutions need >512 tokens
5. **Add curriculum learning** - Start with easier problems

But first: **Get unstuck!** 🚀
