# Training Summary (Step ~3100)

## 1. Critical Issue: "Filibuster" Mode
*   **Observation:** The model is hitting the maximum token limit (`512`) for **100%** of generations (`completions/clipped_ratio: 1.0`).
*   **Symptom:** The `(* END *)` marker is missing. The model generates valid-ish code but fails to stop, likely repeating code or hallucinating endless test cases until cut off.
*   **Impact:** 
    *   **Reward Collapse:** Because the end marker is missing, the "Structural" reward component (0.05) is never awarded.
    *   **Zero Gradient:** `grad_norm` is frequently `0.0`. All 4 generations for a given prompt are often identical (or yield the exact same partial reward), resulting in zero advantage and **no learning**.
    *   **Low Entropy:** Entropy is very low (~0.05), indicating the model is highly confident in this broken strategy.

## 2. Reward Analysis
*   **Current Score:** ~0.21 (stuck).
*   **Likely Breakdown:**
    *   **Type Check (0.20):** The truncated code is somehow passing the type checker (or getting partial credit for 0-1 errors).
    *   **Compilation (0.01):** It gets the "attempted" point.
    *   **Missing:** Structural bonus (0.05) and Test Execution (0.65).
*   **Mystery:** Why does truncated code pass type checking? The `extract_code_block` might be salvaging valid chunks, or the repetition pattern is syntactically valid OCaml.

## 3. Immediate Action Plan
1.  **Resume Training:** (Pending PyTorch version fix/workaround).
2.  **Inspect Logs:** Use the newly enabled `completions.jsonl` to confirm the "filibuster" content (e.g., is it loops? repeated functions?).
3.  **Punish Runaways:** Modify `train.py` to **zero out** all rewards if the completion hits max length and lacks the `(* END *)` marker. This creates a strong gradient against filibustering.
4.  **Boost Exploration:** Increase `GRPO_TEMPERATURE` (0.7 -> 1.0+) to break the "identical generations" cycle.
