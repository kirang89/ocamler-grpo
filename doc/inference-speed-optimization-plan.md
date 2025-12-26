# Lossless Inference Speed Optimizations for GRPO Training

## Goal
Improve inference speed during GRPO training generation phase without any quality loss.

## Target
- **Phase**: Training generation (completions during GRPO)
- **Quality**: Lossless only
- **Constraints**: Latency, throughput, and memory

---

## Optimizations (Priority Order)

### 1. Flash Attention 2 (HIGH IMPACT, LOW COMPLEXITY)

**Expected speedup**: 20-40% faster attention, significant memory reduction

**Changes to `train.py`** (in `create_grpo_config()`):
```python
return GRPOConfig(
    # ... existing params ...
    model_init_kwargs={
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    },
)
```

**Add to `pyproject.toml`**:
```toml
[project.optional-dependencies]
cuda = [
    "torch>=2.6.0",
    "torchvision>=0.21.0",
    "flash-attn>=2.5.0; sys_platform == 'linux'",
]
```

---

### 2. CUDA Optimizations (HIGH IMPACT, TRIVIAL)

**Expected speedup**: 10-20% faster matmuls

**Add to `train.py`** at start of `main()`:
```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
```

---

### 3. vLLM Integration (HIGHEST IMPACT, MEDIUM COMPLEXITY)

**Expected speedup**: 1.4-1.7x for generation

**Important**: Known issue with vLLM colocate + LoRA on multi-GPU ([#3671](https://github.com/huggingface/trl/issues/3671)). Use **server mode** for 3x A40 setup.

**Add to `pyproject.toml`**:
```toml
vllm = [
    "vllm>=0.8.5; sys_platform == 'linux'",
]
```

**Changes to `train.py`** (in `create_grpo_config()`):
```python
use_vllm = os.environ.get("GRPO_USE_VLLM", "false").lower() == "true"
vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "")

return GRPOConfig(
    # ... existing params ...
    use_vllm=use_vllm,
    vllm_server_host=vllm_server_host if vllm_server_host else None,
)
```

**For multi-GPU** (launch vLLM server separately before training):
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --enable-lora
```

---

### 4. torch.compile (MEDIUM IMPACT, LOW COMPLEXITY)

**Expected speedup**: 10-30% faster forward passes (after warmup)

**Changes to `train.py`** (after trainer creation):
```python
use_torch_compile = os.environ.get("GRPO_USE_TORCH_COMPILE", "false").lower() == "true"

trainer = GRPOTrainer(...)

if use_torch_compile:
    trainer.model = torch.compile(trainer.model, mode="reduce-overhead")
```

---

### 5. DataLoader Tuning (LOW IMPACT)

**Optional improvement** - increase workers with 27 vCPUs available:
```python
dataloader_num_workers=4,
dataloader_prefetch_factor=2,
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `train.py:170-238` | Add `model_init_kwargs` with Flash Attention 2 |
| `train.py:284+` | Add CUDA optimizations at start of `main()` |
| `train.py:170-238` | Add vLLM config params |
| `train.py:315+` | Add optional torch.compile after trainer creation |
| `pyproject.toml` | Add `flash-attn` and `vllm` to cuda dependencies |

---

## Environment Variables

```bash
# vLLM (enable when ready)
export GRPO_USE_VLLM=true
export VLLM_SERVER_HOST=localhost:8000  # For server mode

# torch.compile (optional)
export GRPO_USE_TORCH_COMPILE=true
```

---

## Implementation Order

1. **Phase 1 (Quick wins)**: Flash Attention 2 + CUDA optimizations
2. **Phase 2 (vLLM)**: Add vLLM server mode support
3. **Phase 3 (Optional)**: torch.compile support

---

## Verification

- Flash Attention: Check logs for `"flash_attention_2"` during model init
- CUDA TF32: `torch.backends.cuda.matmul.allow_tf32` returns `True`
- vLLM: Look for vLLM init messages; compare generation time per step
- Overall: Track `samples_per_second` metric in training logs

---

## Post-Optimization: Multi-GPU Training with Accelerate

After implementing the optimizations above, further improve training throughput by running on all 3 A40 GPUs with Accelerate.

### 1. Remove ProcessPoolExecutor from Reward Computation

The current `reward.py` uses `ProcessPoolExecutor` for parallel reward computation. With multi-GPU training via Accelerate, this can cause issues with CUDA contexts and is redundant.

**Changes to `reward.py`**:

Replace `_compute_rewards_parallel()` with sequential computation:

```python
# Remove ProcessPoolExecutor usage
# Before:
def _compute_rewards_parallel(completions, ids, tests_list):
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(_score_single, ...))
    return results

# After:
def _compute_rewards_sequential(completions, ids, tests_list):
    results = []
    for completion, problem_id, tests in zip(completions, ids, tests_list):
        result = _score_single(completion, problem_id, tests)
        results.append(result)
    return results
```

**Why**:
- Accelerate handles parallelism across GPUs
- ProcessPoolExecutor can conflict with CUDA contexts in multi-GPU setups
- Reward computation (OCaml compilation) is CPU-bound and runs fine sequentially
- Each GPU process handles its own batch of rewards

### 2. Configure Accelerate for 3x A40 GPUs

**Add to `pyproject.toml`**:
```toml
dependencies = [
    # ... existing deps ...
    "accelerate>=1.2.0",
]
```

**Create `accelerate_config.yaml`**:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 3
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

**Or run interactively**:
```bash
accelerate config
```

### 3. Launch Training with Accelerate

**Instead of**:
```bash
python train.py
```

**Use**:
```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

**Or without config file**:
```bash
accelerate launch --num_processes 3 --mixed_precision bf16 train.py
```

### 4. Adjust Batch Sizes for Multi-GPU

With 3 GPUs, effective batch size = `per_device_batch * 3`:

```bash
# Per-GPU batch of 8 = effective batch of 24
export GRPO_BATCH_SIZE=8
export GRPO_NUM_GENERATIONS=16
export GRPO_GRAD_ACCUM_STEPS=2  # Effective batch = 24 * 2 = 48
```

### Files to Modify for Multi-GPU

| File | Changes |
|------|---------|
| `reward.py` | Remove `ProcessPoolExecutor`, use sequential reward computation |
| `pyproject.toml` | Add `accelerate>=1.2.0` dependency |
| New: `accelerate_config.yaml` | Accelerate configuration for 3x A40 |

### Expected Benefits

- **3x throughput**: Each GPU processes its own batch in parallel
- **Better GPU utilization**: All 3 A40s working simultaneously
- **Simpler reward code**: No ProcessPoolExecutor complexity
- **Automatic gradient sync**: Accelerate handles gradient averaging across GPUs
