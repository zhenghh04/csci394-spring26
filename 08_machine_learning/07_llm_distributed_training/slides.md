# Distributed Training for Large Language Models

**CSCI 394 -- Spring 2026**

---

## 1. Motivation: Why Distributed Training?

### The Problem

Training a **7B-parameter model** with Adam in mixed precision requires:

```
Parameters (fp16):     7B × 2 bytes  =  14 GB
Gradients (fp16):      7B × 2 bytes  =  14 GB
Optimizer state (fp32): 7B × 8 bytes  =  56 GB
Activations:           ~10 GB
────────────────────────────────────
TOTAL:                ~94 GB
```

An A100 GPU has **80 GB** of memory.

**This model does not fit on a single GPU.**

---

## 2. Training vs. Inference Memory

| Component | Inference | Training |
| --------- | --------- | -------- |
| Model weights (fp16) | 14 GB | 14 GB |
| Activations | tiny | ~10 GB |
| Gradients | — | 14 GB |
| Optimizer state | — | 56 GB |
| **Total** | **14 GB** | **94 GB** |

**Training costs 7x more memory than inference.**

This is why you can run inference on a model that you cannot train.

---

## 3. Scale of the Problem

| Model | Parameters | Training memory | A100s needed |
| ----- | ---------- | --------------- | ------------ |
| GPT-2 | 1.5B | 28 GB | 1 |
| LLaMA-7B | 7B | 112 GB | 2 |
| LLaMA-13B | 13B | 226 GB | 3 |
| LLaMA-65B | 65B | 1 TB | 13 |
| GPT-3 | 175B | 2.8 TB | 35 |

Doubling the model size requires more than double the memory.

---

## 4. The Growth of LLM Model Sizes in History

### Exponential Growth of Parameters Over Time

| Year | Model | Parameters | Training data | Key innovation |
| ---- | ----- | ---------- | -------------- | -------------- |
| 2018 | GPT-1 | 117M | 40 GB | Decoder-only, pre-train then fine-tune |
| 2019 | GPT-2 | 1.5B | 40 GB | Scaling up language modeling |
| 2020 | GPT-3 | 175B | 300B tokens | In-context learning, few-shot prompting |
| 2021 | T5-11B | 11B | 750GB | Encoder-decoder, GLUE benchmark |
| 2021 | Jurassic-1 | 178B | 300B tokens | Competitive with GPT-3 |
| 2022 | LLaMA-1 | 65B | 1.4T tokens | Efficient training, open-weight |
| 2023 | LLaMA-2 | 70B | 2T tokens | Improved instructions, aligned |
| 2023 | Mixtral-8x7B | 56B | 2T tokens | Mixture of Experts (MoE) |
| 2024 | LLaMA-3 | 70B | 15T tokens | Better scaling, longer context |
| 2024 | GPT-4 | 1.8T+ | proprietary | Multimodal, reasoning |
| 2025 | DeepSeek-R1 | 671B | multi-trillion | Extended thinking, reasoning |

### Key Trends

1. **Parameter growth**: 117M → 1.8T+ in 7 years (**~15,000x**)
2. **Training data**: 40 GB → 15+ trillion tokens (**~375,000x**)
3. **Compute cost**: Exponential — single models now cost **$5M--$100M+** to train
4. **Efficiency shift**: Recent models (LLaMA, Mixtral) are more parameter-efficient than early giants

---

## 5. Training Time and Cost

### Real Training Runs

| Model | Tokens | GPU-hours | Calendar time | Cost |
| ----- | ------ | --------- | ------------- | ---- |
| LLaMA-7B | 1T | 35K | ~35 hours | $100K |
| LLaMA-65B | 1.4T | 82K | ~82 hours | $250K |
| GPT-3 (175B) | 300B | 3.6M | 150 days | $5--10M |

**Training at scale is measured in millions of GPU-hours and tens of millions of dollars.**

---

## 6. FLOPs: The Math Behind Training Time

### Training FLOPs Formula

For a model with `N` parameters trained on `D` tokens:

```
Total FLOPs ≈ 6 × N × D
```

The factor of 6 comes from:
- 1× for forward pass
- 2× for backward pass (weight gradients + activation gradients)
- 2× more for multiply-accumulate structure

### Chinchilla Scaling Law

For a given compute budget `C`, the compute-optimal model size is:

```
N_opt ≈ 0.2 × C^0.5
D_opt ≈ 20 × N
```

**Key insight:** Most early LLMs (GPT-3) were **over-parameterized and under-trained** relative to their compute budget.

---

## 7. GPU Utilization: The Roofline Model

Not all GPUs run at peak speed. The key metric is **Model FLOPs Utilization (MFU)**:

```
MFU = (actual FLOPs/s) / (theoretical peak FLOPs/s)
```

### Real MFU in Practice

| Setup | MFU |
| ----- | --- |
| Naive PyTorch, single GPU | 10--25% |
| Flash Attention + mixed precision | 35--45% |
| Production (Megatron-LM) | 45--60% |

**The gap is communication and memory bandwidth, not compute.**

---

## 8. Why Data Parallelism Alone is Not Enough

### Data Parallelism (DDP)

Every GPU holds a **full copy** of the model. Only the data is split.

```
GPU 0:  [full model] ──→ process batch [0..63]
GPU 1:  [full model] ──→ process batch [64..127]
GPU 2:  [full model] ──→ process batch [128..191]
GPU 3:  [full model] ──→ process batch [192..255]

After backward:  all-reduce gradients
```

### The Problem

- If the model requires 112 GB and each GPU has 80 GB, **DDP cannot start.**
- Adding more GPUs does **not help** with the memory problem.
- Each GPU still needs the full model.

**DDP solves throughput, not capacity.**

---

## 9. The Four Parallelism Strategies

To solve the **memory problem**, we need to split the model itself.

| Strategy | What's split | Main benefit | Main cost |
| -------- | ------------ | ------------ | --------- |
| Data Parallelism | Training data | Throughput | Gradient communication |
| Tensor Parallelism | Tensors inside a layer | Fits models on 1 node | High intra-layer communication |
| Pipeline Parallelism | Model layers | Fits models across nodes | Pipeline bubbles, scheduling |
| ZeRO / FSDP | Model state (params, grads, optimizer) | Memory efficiency | Frequent gather/scatter |

---

## 10. Tensor Parallelism (TP): Column Parallel

### The Idea

Split a weight matrix by **output features (columns)**.

```
Full weight W: shape [out_features, in_features]

GPU 0: W₀ = W[0        : out/2, :]    Y₀ = X @ W₀ᵀ
GPU 1: W₁ = W[out/2    : out,   :]    Y₁ = X @ W₁ᵀ

Combine: all-gather(Y₀, Y₁) → full output Y = [Y₀ | Y₁]
```

### Communication

**All-gather**: Each GPU sends its local output to all others.

---

## 11. Tensor Parallelism (TP): Row Parallel

### The Idea

Split a weight matrix by **input features (rows)**.

```
Full input X: shape [batch, in]
Full weight W: shape [out, in]

GPU 0: X₀ = X[:, 0    : in/2],  W₀ = W[:, 0    : in/2]  →  partial sum S₀
GPU 1: X₁ = X[:, in/2 : in  ],  W₁ = W[:, in/2 : in  ]  →  partial sum S₁

Combine: all-reduce(S₀ + S₁) → full output Y = S₀ + S₁
```

### Communication

**All-reduce**: Sum partial results and broadcast the full result.

---

## 12. Megatron-LM Pattern: Combining TP

### Efficient Use of TP in Transformers

Column-parallel → Row-parallel (with all-reduce at layer boundary)

```
Input X
    ↓
[Column Parallel]  →  all-gather
    ↓
FFN middle (no communication)
    ↓
[Row Parallel]  →  all-reduce
    ↓
Output
```

### Why TP Must Stay Within a Node

Tensor parallelism requires communication on **every forward and backward pass**.

- Within a node (NVLink): 600 GB/s bidirectional ✓
- Across nodes (InfiniBand): 400 GB/s ✗ (communication dominates)

**Tensor parallelism is kept intra-node. Inter-node requires other strategies.**

---

## 13. Pipeline Parallelism (PP): Splitting Model Depth

### The Idea

Divide the model's **layers** across different GPUs or nodes.

```
Stage 0 (GPU 0): layers 1--12    (embedding + 12 transformer blocks)
Stage 1 (GPU 1): layers 13--24
Stage 2 (GPU 2): layers 25--36
Stage 3 (GPU 3): layers 37--48   (+ LM head)
```

Data flows through stages like an **assembly line**.

---

## 14. Pipeline Parallelism: Micro-batches

### Forward and Backward Waves

To keep all stages busy, split the global batch into **micro-batches**.

```
Time  →

Stage 0:  [mb1-F]  [mb2-F]  [mb3-F]  [mb4-F]  .  .  [mb4-B]  [mb3-B]
Stage 1:   .      [mb1-F]  [mb2-F]  [mb3-F]  [mb4-F] [mb4-B]  [mb3-B]
Stage 2:   .       .      [mb1-F]  [mb2-F]  [mb3-F] [mb4-B]
Stage 3:   .       .       .      [mb1-F]  [mb2-F] [mb1-B]
```

**Pipeline bubble**: stages are idle during startup and drain.

---

## 15. Pipeline Bubble and Efficiency

### Bubble Formula

```
Pipeline bubble = (p - 1) / (m + p - 1)

where:
  p = number of pipeline stages
  m = number of micro-batches
```

### Examples

- `p = 4 stages, m = 8 micro-batches`: bubble = 3/11 ≈ **27%**
- `p = 4 stages, m = 32 micro-batches`: bubble = 3/35 ≈ **9%**

**More micro-batches reduce bubble but increase activation memory.**

---

## 16. Pipeline Parallelism: 1F1B Schedule

### The Problem

With naive scheduling, stages are idle while waiting for data.

### The Solution: 1 Forward 1 Backward (1F1B)

Interleave forward and backward passes strategically:

```
Stage 0:  [1F]  [2F]  [3F]  [4F]  [4B]  [3B]  [2B]  [1B]
Stage 1:   .   [1F]  [2F]  [3F]  [3B]  [2B]  [1B]
Stage 2:   .    .   [1F]  [2F]  [2B]  [1B]
Stage 3:   .    .    .   [1F]  [1B]
```

This reduces the bubble significantly with the same memory.

---

## 17. ZeRO Stage 1: Shard Optimizer State

### The Problem with DDP

Every GPU holds:
- Full parameters (16 bytes/param)
- Full gradients (2 bytes/param)
- **Full optimizer state (8 bytes/param)** ← redundant!

With 8 GPUs, optimizer state is replicated 8× unnecessarily.

### ZeRO Stage 1 Solution

Each GPU holds only **1/N of the optimizer states** (Adam first and second moments, master weights).

```
Memory saving: eliminate ~75% of optimizer state redundancy
```

---

## 18. ZeRO Stage 2: Shard Gradients Too

### The Problem

Even with Stage 1, gradients are still fully replicated.

### ZeRO Stage 2 Solution

Each GPU holds full parameters, but only **1/N of the gradients and optimizer states**.

Training loop:
1. **Reduce-scatter**: Each GPU keeps its shard of gradients
2. **Optimizer step**: Each GPU updates only its shard
3. **All-gather**: Reconstruct full parameters for next forward pass

```
Memory saving: eliminate ~87% of per-GPU redundancy
```

---

## 19. ZeRO Stage 3: Shard Parameters Too

### The Ultimate Savings

Shard all parameters, all gradients, all optimizer states.

**Before each forward pass**: all-gather layer parameters as they are needed  
**After forward/backward**: discard non-owner shards

```
Memory per GPU = (Total model state) / (world_size)
```

### Example: GPT-3 (175B) on 64 GPUs

| Strategy | Memory per GPU |
| -------- | -------------- |
| DDP | 2.8 TB ✗ |
| ZeRO Stage 1 | ~0.8 TB |
| ZeRO Stage 2 | ~0.35 TB |
| ZeRO Stage 3 | **44 GB ✓** |

**ZeRO Stage 3 makes GPT-3-scale training possible.**

---

## 20. FSDP: PyTorch's ZeRO Implementation

### What is FSDP?

**Fully Sharded Data Parallel** is PyTorch's built-in implementation of ZeRO Stage 3.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)
```

### How it Works

1. **At rest**: Parameters are sharded across GPUs
2. **Forward pass**: All-gather full parameters into layer
3. **Backward pass**: All-gather gradients, reduce-scatter to shards
4. **Optimizer step**: Local update on each shard

---

## 21. Communication Patterns Summary

| Strategy | Collective | When | Bandwidth need |
| -------- | ---------- | ---- | -------------- |
| Data Parallelism | all-reduce | after backward | medium |
| Tensor Parallelism | all-gather + all-reduce | every layer | **very high** |
| Pipeline Parallelism | send/recv | stage boundaries | medium |
| ZeRO Stage 2 | reduce-scatter | after backward | medium |
| ZeRO Stage 3 | all-gather + reduce-scatter | every module | high |

---

## 22. 3D Parallelism: Composing All Strategies

### Real Production Systems Combine All Four

```
World = (D × T × P) GPUs

D  data-parallel groups     (different token batches)
T  tensor-parallel groups   (within a node via NVLink)
P  pipeline stages          (across nodes)
```

### Example: 530B-parameter Model on 2048 GPUs

```
Pipeline stages:  P = 8   (each holds ~66B params)
Tensor parallel:  T = 8   (within one 8-GPU node)
Data parallel:    D = 32  (32 independent model replicas)

Total: 8 × 8 × 32 = 2048 GPUs
```

Each pipeline stage is a tensor-parallel group, and there are 32 such groups in data parallelism.

---

## 23. Decision Tree: Which Strategy to Use?

```
Does the model fit on one GPU?
├─ YES  → Use DDP only (simple + efficient)
└─ NO   → Does it fit with ZeRO Stage 3?
         ├─ YES  → Use FSDP (PyTorch built-in)
         └─ NO   → Need pipeline parallelism
                   └─ Add tensor parallelism within nodes
                      → Use Megatron-LM or similar
```

### Quick Reference

| Situation | Strategy |
| --------- | -------- |
| Single GPU training | No distributed needed |
| Model fits with ZeRO-3 | FSDP + DDP |
| Very large model (>100B) | TP + PP + DP (3D) |
| Very long sequences | TP + sequence parallelism |
| Maximum performance | Megatron-LM or equivalents |

---

## 24. Key Takeaways

### Why Distributed Training?

- Training memory is **8x inference memory**
- Large models simply do not fit on one GPU
- Data parallelism alone helps throughput, not capacity

### Four Strategies

1. **Data Parallelism**: Split data, replicate model
2. **Tensor Parallelism**: Split layers, keep within node
3. **Pipeline Parallelism**: Split model depth across nodes
4. **ZeRO / FSDP**: Shard model state for memory efficiency

### Production Reality

Real LLM training uses **all four simultaneously** to balance:
- Memory requirements
- Compute throughput
- Communication efficiency
- Implementation complexity

---

## 25. The Numbers You Should Remember

| Metric | Value |
| ------ | ----- |
| Training memory per parameter (Adam, fp16) | 16 bytes |
| Training / inference memory ratio | ~8x |
| LLaMA-7B total training memory | 112 GB |
| A100 GPU memory | 80 GB |
| GPUs needed for 7B training (minimum) | 2 |
| Model FLOPs Utilization (production) | 40--60% |
| GPT-3 total training FLOPs | 3.15×10²³ |
| GPT-3 training cost | $5--10M |

---

## 26. Hands-On Demo: DDP for LLMs

### Single Process (CPU)

```bash
python3 train_tiny_llm_ddp.py --cpu --epochs 10
```

### Multi-Process with DDP (2 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 \
  train_tiny_llm_ddp.py --epochs 10
```

### Tensor Parallel Demo (2 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 \
  tensor_parallel_linear_demo.py
```

---

## 27. Exercises

### Easy

1. A 13B-parameter model with AdamW in fp16 requires how much memory?
   Does it fit on 2 × A100-80GB?

2. Why is DDP scaling usually worse across nodes than within a node?

### Medium

3. With pipeline parallelism on 4 stages and 16 micro-batches, what is the
   pipeline bubble fraction?

4. In ZeRO Stage 3 with 64 GPUs, how much memory does each GPU need for a
   7B-parameter model (16 bytes/param)?

### Hard

5. Suppose you have 256 A100 GPUs for 30 days with 40% MFU. Using the
   formula `6 × N × D`, how many tokens can you train a 7B model on?

6. Design a 3D parallelism strategy (D, T, P) for 1024 GPUs to train a
   100B-parameter model, minimizing per-GPU memory.

---

## 28. References and Further Reading

### Papers

- **Attention Is All You Need** (Vaswani et al., 2017)
  - Transformer architecture
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
  (Shoeybi et al., 2019)
  - Tensor and pipeline parallelism for LLMs
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**
  (Rajbhandari et al., 2020)
  - Sharded model state
- **Efficient Large-Scale Language Modeling with Mixtures of Experts**
  (Lepikhin et al., 2021)
  - Modern scaling strategies

### Code

- `train_tiny_llm_ddp.py` -- Runnable DDP demo
- `tensor_parallel_linear_demo.py` -- Tensor parallelism visualization
- PyTorch FSDP -- `torch.distributed.fsdp`
- Megatron-LM -- `nvidia/Megatron-LM` on GitHub

---

## 29. Summary Table: All Strategies at a Glance

| Strategy | Memory savings | Communication | Complexity | When to use |
| -------- | -------------- | -------------- | ---------- | ----------- |
| DDP | None | per-step all-reduce | Low | Model fits 1 GPU |
| TP | 1/T (intra-layer) | per-layer collectives | Medium | Within fast node |
| PP | N/P (per stage) | point-to-point | Medium | Deep models |
| ZeRO-1 | Optimizer state 75% | per-step | Low | Initial scaling |
| ZeRO-2 | State + grads 87% | per-step | Medium | Better memory |
| ZeRO-3 | All state (N×) | per-module | High | Very large models |

---

## The End

**Distributed LLM training is an HPC problem.**

Solve it by carefully composing compute, memory, and communication strategies to train the largest models in the shortest time with available hardware.

Questions?
