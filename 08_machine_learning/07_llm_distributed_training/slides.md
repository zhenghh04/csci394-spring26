# Distributed Training for Large Language Models

**CSCI 394 -- Spring 2026**

---

## Motivation: One Training Step Is an HPC Job

![LLM training systems loop](figures/hpc_training_loop.svg){width=88%}

---

## 1. Motivation: Why Distributed Training?

![Memory wall for 7B training](figures/memory_wall_7b.svg){width=88%}

**The first lesson:** distributed training is not only about speed. It is often
the only way to make the model fit at all.

---

## 2. Training vs. Inference Memory

| Component | Inference | Training |
| --------- | --------- | -------- |
| Model weights (fp16) | 14 GB | 14 GB |
| Activations | tiny | ~10 GB |
| Gradients | — | 14 GB |
| Adam moments + master weights | — | 84 GB |
| **Total** | **14 GB** | **122 GB** |

Model state alone costs **8x inference memory** before activations.

This is why you can run inference on a model that you cannot train.

---

## 3. Scale of the Problem

| Model | Parameters | Training memory | A100-80GBs needed |
| ----- | ---------- | --------------- | ------------------ |
| GPT-2 XL | 1.5B | ~28 GB | 1 |
| LLaMA-7B | 7B | ~122 GB | 2 |
| LLaMA-13B | 13B | ~226 GB | 3 |
| LLaMA-65B | 65B | ~1.1 TB | 14 |
| GPT-3 | 175B | ~3 TB | 38 |

These are lower bounds: temporary buffers, fragmentation, and longer sequences
increase the real requirement.

---

## 4. The Growth of LLM Model Sizes

![LLM model size timeline](figures/llm_scaling_timeline.svg){width=88%}

### Key Trends

1. **Parameter growth**: millions to hundreds of billions in a few years
2. **Training data growth**: billions to trillions of tokens
3. **Compute cost**: single training runs can cost millions of GPU-hours
4. **Efficiency shift**: newer models often spend more tokens per parameter

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
- about `2 × N` FLOPs per token for the forward pass
- backward is about 2x the forward pass
- training is forward + backward: `3 × 2 × N × D`

### Chinchilla Scaling Law

For a given compute budget `C`, the compute-optimal model size is:

```
N_opt ≈ 0.2 × C^0.5
D_opt ≈ 20 × N
```

**Key insight:** Most early LLMs (GPT-3) were **over-parameterized and under-trained** relative to their compute budget.

---

## 7. GPU Utilization: The Roofline Model

![Roofline model for LLM training MFU](figures/roofline_mfu.svg){width=88%}

Not all GPUs run at peak speed. The key metric is **Model FLOPs Utilization (MFU)**:

```
MFU = (actual FLOPs/s) / (theoretical peak FLOPs/s)
```

**The gap is communication and memory bandwidth, not compute.**

---

## 8. Why Data Parallelism Alone is Not Enough

![Data parallelism throughput and capacity limit](figures/data_parallel_limit.svg){width=88%}

### The Problem

- If the model requires 122 GB and each GPU has 80 GB, **DDP cannot start.**
- Adding more GPUs does **not help** with the memory problem.
- Each GPU still needs the full model.

**DDP solves throughput, not capacity.**

---

## 9. The Four Parallelism Strategies

To solve the **memory problem**, we need to split the model itself.

![Four parallelism strategies](figures/parallelism_map.svg){width=88%}

The art is choosing which dimension to split without making communication the
new bottleneck.

---

## 10. Tensor Parallelism (TP): Column Parallel

![Tensor parallel linear layer schematic](figures/tensor_parallel_schematic.svg){width=82%}

- **Column parallel**: split output features, then all-gather the output.
- **Row parallel**: split input features, then all-reduce partial sums.

---

## 11. Tensor Parallelism (TP): Row Parallel

### The Idea

Split a weight matrix by **input features (rows)**.

```
Full input X: shape [batch, in]
Full weight W: shape [in, out]

GPU 0: X0 = X[:, 0    : in/2],  W0 = W[0    : in/2, :]  ->  partial S0
GPU 1: X1 = X[:, in/2 : in  ],  W1 = W[in/2 : in,   :]  ->  partial S1

Combine: all-reduce(S0 + S1) -> full output Y = S0 + S1
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

- Within a node (NVLink): very high bandwidth and low latency
- Across nodes (InfiniBand): useful, but latency makes per-layer collectives expensive

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

To keep all stages busy, split the global batch into **micro-batches**.

![Pipeline parallel schedule and bubble](figures/pipeline_schedule_bubble.svg){width=88%}

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
- Full parameters (2 bytes/param)
- Full gradients (2 bytes/param)
- **Full Adam state + master weights (12 bytes/param)** -- redundant!

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

![ZeRO stages memory savings](figures/zero_stages_memory.svg){width=82%}

Stage 3 shards **parameters, gradients, and optimizer state**. During forward
and backward, each layer gathers what it needs and discards it afterward.

For GPT-3 model state on 64 GPUs, Stage 3 reduces per-GPU state from
**2.8 TB** to about **44 GB** before activations.

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

![3D parallelism cube](figures/parallelism_cube_3d.svg){width=88%}

### Example: 530B-parameter Model on 2048 GPUs

```
Pipeline stages:  P = 8   (each holds ~66B params)
Tensor parallel:  T = 8   (within one 8-GPU node)
Data parallel:    D = 32  (32 independent model replicas)

Total: 8 x 8 x 32 = 2048 GPUs
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

- Training model state is **~8x inference weights**
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

## Why This Should Matter to You

This lecture is not only about foundation-model companies.

- The same ideas appear in scientific ML, climate models, protein models,
  recommender systems, and multimodal models.
- A 10% utilization improvement on 1,000 GPUs is a large engineering win.
- Students who understand **memory + compute + communication** can reason about
  systems that most users only treat as black boxes.

**The skill is not "use more GPUs." The skill is knowing what to split.**

---

## 25. The Numbers You Should Remember

| Metric | Value |
| ------ | ----- |
| Training memory per parameter (Adam, fp16) | 16 bytes |
| Training / inference model-state ratio | ~8x |
| LLaMA-7B model state + activations | ~122 GB |
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
- `build_figures.py` -- Regenerates the schematic SVG figures
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
