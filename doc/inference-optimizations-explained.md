# Inference Speed Optimizations Explained

## 1. Flash Attention 2

### What is Attention?

In transformers, attention computes how much each token should "pay attention" to every other token. For a sequence of length `N`, this requires:

```
Q × K^T → N × N attention matrix → softmax → multiply by V
```

**The problem**: The `N × N` attention matrix must be stored in GPU memory (HBM - High Bandwidth Memory). For a 2048-token sequence, that's 4 million elements per attention head.

### What Flash Attention Does

Flash Attention restructures the computation to avoid materializing the full `N × N` matrix:

```
Standard Attention:
┌─────────────────────────────────────────────────────┐
│ GPU HBM (slow, large)                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │ Q,K,V   │───>│ N×N mat │───>│ Output  │         │
│  └─────────┘    └─────────┘    └─────────┘         │
│                 (bottleneck)                        │
└─────────────────────────────────────────────────────┘

Flash Attention:
┌─────────────────────────────────────────────────────┐
│ GPU HBM                                             │
│  ┌─────────┐                   ┌─────────┐         │
│  │ Q,K,V   │                   │ Output  │         │
│  └────┬────┘                   └────▲────┘         │
│       │                             │               │
│  ┌────▼─────────────────────────────┴────┐         │
│  │ GPU SRAM (fast, small)                │         │
│  │  Process in tiles, never store full   │         │
│  │  N×N matrix                           │         │
│  └───────────────────────────────────────┘         │
└─────────────────────────────────────────────────────┘
```

**Key insight**: GPU SRAM is ~10-20x faster than HBM but much smaller. Flash Attention computes attention in small tiles that fit in SRAM, using a numerically-stable online softmax algorithm.

### Why It Helps

| Benefit | Reason |
|---------|--------|
| **Memory**: O(N) instead of O(N²) | No full attention matrix stored |
| **Speed**: 2-4x faster | Fewer slow HBM reads/writes |
| **Longer sequences** | Can handle 16k+ tokens without OOM |

---

## 2. TF32 (TensorFloat-32)

### What is Floating Point?

Numbers in GPUs are stored as floating point with limited precision:

```
FP32 (32 bits): 1 sign + 8 exponent + 23 mantissa bits
FP16 (16 bits): 1 sign + 5 exponent + 10 mantissa bits
BF16 (16 bits): 1 sign + 8 exponent + 7 mantissa bits
TF32 (19 bits): 1 sign + 8 exponent + 10 mantissa bits (internal only)
```

### What TF32 Does

TF32 is a **compute format** (not storage) used internally by Tensor Cores on Ampere+ GPUs:

```
Input (FP32)                    Output (FP32)
┌────────────────┐              ┌────────────────┐
│ 32-bit floats  │              │ 32-bit floats  │
└───────┬────────┘              └───────▲────────┘
        │                               │
        ▼                               │
┌───────────────────────────────────────┴───────┐
│              Tensor Core                       │
│  ┌─────────────────────────────────────────┐  │
│  │ Truncate mantissa: 23 bits → 10 bits    │  │
│  │ Compute matrix multiply in TF32         │  │
│  │ Accumulate result in FP32               │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

**Key insight**: Most neural network operations don't need 23 bits of mantissa precision. By using 10 bits internally, Tensor Cores can do 8x more operations per cycle.

### Why It Helps

| Aspect | FP32 | TF32 |
|--------|------|------|
| Storage precision | 23 bits | 23 bits (unchanged) |
| Compute precision | 23 bits | 10 bits |
| Tensor Core throughput | 19.5 TFLOPS | 156 TFLOPS (A40) |
| Accuracy loss | None | Negligible for training |

The model weights and gradients are still stored in FP32, only the matrix multiply uses reduced precision.

---

## 3. cuDNN Benchmark Mode

### What is cuDNN?

cuDNN is NVIDIA's library for neural network primitives (convolutions, attention, etc.). For each operation, cuDNN has multiple algorithms:

```
Convolution algorithms for a given input shape:
┌────────────────────────────────────────────────┐
│ Algorithm 0: IMPLICIT_GEMM        - 2.3 ms     │
│ Algorithm 1: IMPLICIT_PRECOMP     - 1.8 ms     │
│ Algorithm 2: GEMM                 - 3.1 ms     │
│ Algorithm 3: FFT                  - 1.2 ms  <-- fastest for this shape
│ Algorithm 4: WINOGRAD             - 1.5 ms     │
└────────────────────────────────────────────────┘
```

The optimal algorithm depends on:
- Input tensor shape
- GPU architecture
- Available memory

### What Benchmark Mode Does

```
First forward pass (with benchmark=True):
┌─────────────────────────────────────────────┐
│ For each layer:                             │
│   1. Try all algorithms                     │
│   2. Time each one                          │
│   3. Cache the fastest                      │
│                                             │
│ Layer 1: FFT wins (cached)                  │
│ Layer 2: WINOGRAD wins (cached)             │
│ Layer 3: IMPLICIT_GEMM wins (cached)        │
└─────────────────────────────────────────────┘

Subsequent passes:
┌─────────────────────────────────────────────┐
│ Use cached optimal algorithms               │
│ No benchmarking overhead                    │
└─────────────────────────────────────────────┘
```

### Why It Helps

| Without Benchmark | With Benchmark |
|-------------------|----------------|
| Uses default algorithm | Uses fastest algorithm |
| Consistent but suboptimal | First batch slower, then faster |
| Works with variable shapes | Best for fixed shapes (training) |

**Caveat**: Only helps when input shapes are consistent (same batch size, sequence length). GRPO training has consistent shapes, so this is safe.

---

## 4. torch.compile

### What is torch.compile?

PyTorch normally executes operations eagerly (one at a time):

```python
# Eager execution
x = input @ weight1      # GPU kernel 1 -> wait
x = x + bias1            # GPU kernel 2 -> wait
x = torch.relu(x)        # GPU kernel 3 -> wait
x = x @ weight2          # GPU kernel 4 -> wait
```

Each operation launches a GPU kernel, waits for completion, then launches the next. This has overhead:
- CPU->GPU launch latency (~5-10 us per kernel)
- GPU sits idle between kernels

### What torch.compile Does

```
┌─────────────────────────────────────────────────────┐
│ torch.compile process:                              │
│                                                     │
│ 1. Trace Python code to capture computation graph  │
│ 2. Optimize graph (fuse operations, eliminate dead │
│    code, reorder for memory locality)              │
│ 3. Generate optimized CUDA kernels                 │
│ 4. Cache compiled code for reuse                   │
└─────────────────────────────────────────────────────┘

Before (4 kernels):
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│matmul  │>│  add   │>│  relu  │>│matmul  │
└────────┘ └────────┘ └────────┘ └────────┘
   | wait    | wait    | wait    | wait

After fusion (2 kernels):
┌─────────────────────────┐ ┌────────┐
│ matmul + add + relu     │>│matmul  │
│ (fused kernel)          │ │        │
└─────────────────────────┘ └────────┘
         | wait               | wait
```

### Why It Helps

| Optimization | Benefit |
|--------------|---------|
| **Kernel fusion** | Fewer kernel launches, less memory traffic |
| **Memory planning** | Reuse memory buffers, reduce allocations |
| **Operator reordering** | Better cache locality |
| **Dead code elimination** | Skip unnecessary computations |

### Trade-offs

| Pro | Con |
|-----|-----|
| 10-30% faster inference | First run is slow (compilation) |
| Works with dynamic shapes (with guards) | Can have "graph breaks" on unsupported ops |
| No code changes needed | Debugging is harder |

The `mode="reduce-overhead"` setting prioritizes reducing kernel launch overhead, which helps most for smaller batch sizes where launch latency dominates.

---

## Summary: Where Time Goes

```
Training step breakdown (before optimization):
┌────────────────────────────────────────────────────────┐
│ Attention computation         ████████████████  40%    │ <- Flash Attention 2
│ Matrix multiplies (forward)   ████████████      30%    │ <- TF32
│ Matrix multiplies (backward)  ████████          20%    │ <- TF32
│ Kernel launch overhead        ███               7%     │ <- torch.compile
│ Algorithm selection           █                 3%     │ <- cuDNN benchmark
└────────────────────────────────────────────────────────┘

After optimization:
┌────────────────────────────────────────────────────────┐
│ Attention computation         ████████          20%    │
│ Matrix multiplies (forward)   ██████            15%    │
│ Matrix multiplies (backward)  ████              10%    │
│ Kernel launch overhead        █                 3%     │
│ Algorithm selection           (cached)          0%     │
│ Other (memory, data loading)  ████████████████  52%    │
└────────────────────────────────────────────────────────┘

Net speedup: ~40-60% faster per training step
```
