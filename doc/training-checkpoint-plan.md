# Plan: Enable Automatic Checkpoint Resumption

## Objective
Modify `train.py` to automatically detect the latest training checkpoint in `GRPO_OUTPUT_DIR` and resume training from that point if the process is restarted. This ensures that training progress (steps/epochs) is not lost if the script is interrupted.

## Rationale
The `GRPOTrainer` (inheriting from Hugging Face's `Trainer`) supports the `resume_from_checkpoint` argument. By using the `get_last_checkpoint` utility, we can dynamically find the most recent state and pass it to the training loop.

## Implementation Steps

### 1. Import Utility
Add the following import to `train.py`:
```python
from transformers.trainer_utils import get_last_checkpoint
```

### 2. Update `main()` function
Modify the execution logic in `main()` to:
1.  Check `GRPO_OUTPUT_DIR` for existing checkpoints using `get_last_checkpoint`.
2.  Log the status (Resuming from `...` or Starting fresh).
3.  Pass the detected checkpoint path to the `train()` method.

**Code Change:**
```python
    # ... inside main() ...
    
    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(GRPO_OUTPUT_DIR)
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    trainer = GRPOTrainer(...)
    
    # Pass resume_from_checkpoint argument
    trainer.train(resume_from_checkpoint=last_checkpoint)
```

## Verification
1.  We will verify that `transformers.trainer_utils` is available (it is part of the standard `transformers` library used in this project).
2.  The `resume_from_checkpoint` argument handles loading the model weights, optimizer state, and scheduler state, ensuring exact continuation.