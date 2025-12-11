# Plan: Add Multiplier Penalty for Runaway Completions

## Problem
Model is hitting max token limit (512) on 100% of generations without producing the `(* END *)` marker, causing:
- Zero gradient (all generations identical → same reward → zero advantage)
- Reward stuck at ~0.21
- No learning signal

## Solution
Apply a **multiplicative penalty** to preserve relative quality ranking while discouraging runaway behavior.

## Implementation

### 1. Add Configuration Constant
**File:** `train.py` (top of file, near other constants)

```python
# Penalty multiplier for completions that hit max length without END marker
# Set to 0.2 (20% of original reward) to strongly discourage filibustering
# while preserving relative quality signals between generations
RUNAWAY_PENALTY_MULTIPLIER = float(os.environ.get("RUNAWAY_PENALTY_MULTIPLIER", "0.2"))
```

**Why:**
- Makes it configurable via environment variable
- Default 0.2 (20%) provides strong discouragement
- Documented inline for clarity

---

### 2. Detect Runaway Completions
**File:** `train.py` in `make_syntax_aware_reward()` function

**Location:** After computing `total_reward` (around line 473), before appending to rewards list

**Add this logic:**
```python
# === Final Reward ===
total_reward = structural_score + type_score + compile_score + test_score

# Apply penalty for runaway completions (hit max length without END marker)
is_runaway = len(completion) >= 500 and not completion.strip().endswith(END_MARKER)
if is_runaway:
    total_reward *= RUNAWAY_PENALTY_MULTIPLIER
    penalty_applied = True
else:
    penalty_applied = False

rewards.append(total_reward)
```

**Why:**
- Uses length threshold (500 chars ≈ near max tokens) + missing END marker
- Multiplicative penalty preserves ranking (0.21 → 0.042, 0.16 → 0.032, etc.)
- Simple boolean flag for logging

---

### 3. Update Logging
**File:** `train.py` in `make_syntax_aware_reward()` function

**Location:** Update the `detailed_logs.append()` call (around line 477)

**Add penalty tracking:**
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
        "runaway_penalty_applied": penalty_applied,  # NEW
        "preview": completion[:200],
    }
)
```

**Why:**
- Track when penalty is applied for analysis
- Helps debug if penalty is too aggressive/lenient

---

### 4. Update Completion Logs
**File:** `train.py` in `make_syntax_aware_reward()` function

**Location:** Update the `completion_logs.append()` call (around line 491)

**Add penalty tracking:**
```python
completion_logs.append(
    {
        "problem_id": pid,
        "reward": float(total_reward),
        "length": len(completion),
        "runaway_penalty_applied": penalty_applied,  # NEW
        "completion": completion,
    }
)
```

**Why:**
- Links penalty flag to full completion text for inspection

---

## Testing

### Before Training
1. **Verify the change doesn't break existing code:**
   ```bash
   python train.py --help  # Check it loads without errors
   ```

2. **Test with different multiplier values:**
   ```bash
   export RUNAWAY_PENALTY_MULTIPLIER=0.1  # More aggressive
   export RUNAWAY_PENALTY_MULTIPLIER=0.5  # More lenient
   ```

### During Training
1. **Monitor logs:** Check `reward_logs/syntax_aware_breakdown.jsonl` for:
   - `runaway_penalty_applied: true` entries
   - Reward distributions before/after penalty

2. **Watch for gradient:** Check training logs for non-zero `grad_norm`

3. **Track clipped ratio:** Should decrease from 1.0 over time

---

## Expected Outcome

**Before:**
- All 4 generations hit max length → rewards: [0.21, 0.21, 0.21, 0.21]
- Advantages: [0, 0, 0, 0] → **Zero gradient**

**After (with 0.2 multiplier):**
- All 4 generations hit max length → raw rewards: [0.21, 0.18, 0.15, 0.12]
- Penalized rewards: [0.042, 0.036, 0.030, 0.024]
- Advantages: [+0.009, +0.003, -0.003, -0.009] → **Non-zero gradient!**

**Long-term:**
- Model learns to avoid filibustering (strong penalty)
- Model still learns which filibuster attempts were better (preserved ranking)
- Enables escape from current local minimum

---

## Rollback Plan
If the penalty causes issues (training instability, reward collapse):

1. **Disable:** `export RUNAWAY_PENALTY_MULTIPLIER=1.0` (no penalty)
2. **Reduce:** Try 0.5 or 0.3 (less aggressive)
3. **Revert code:** Remove the penalty logic entirely

---

## Summary of Changes
- **1 constant added** (configurable via env var)
- **~10 lines of logic** (penalty calculation + detection)
- **2 logging fields added** (for tracking)
- **Zero breaking changes** (existing behavior preserved when multiplier = 1.0)
