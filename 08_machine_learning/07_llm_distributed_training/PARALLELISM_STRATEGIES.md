# Parallelism Strategies for LLM Training

This note covers the four main techniques used to distribute LLM training
across many accelerators:

1. **Data parallelism** -- split the data, replicate the model
2. **Tensor parallelism** -- split a layer's math across GPUs
3. **Pipeline parallelism** -- split the model's depth across GPUs
4. **ZeRO / FSDP** -- shard model state to reduce memory per GPU

In practice, production systems combine all four simultaneously. The goal of
this note is to understand each strategy independently before seeing them
composed.

---

## 1. Data Parallelism (DP)

### The idea

Every GPU holds a **full copy** of the model. The training dataset is
partitioned so each GPU sees different batches. After the backward pass,
gradients are averaged across all GPUs so the models stay identical.

```
GPU 0:  model (full copy)  |  batch [0..63]
GPU 1:  model (full copy)  |  batch [64..127]
GPU 2:  model (full copy)  |  batch [128..191]
GPU 3:  model (full copy)  |  batch [192..255]

After backward:  all-reduce gradients  -->  identical model updates on all GPUs
```

### Communication

The collective is **all-reduce** on gradients: each GPU sums its local
gradients with the gradients from all other GPUs.

In PyTorch DDP, the all-reduce overlaps with the backward pass (gradient
buckets are communicated as they are computed, not all at once at the end).

### Effective batch size

The global batch size is `local_batch_size × world_size`. Larger global batch
sizes often require adjusting the learning rate (linear scaling rule) and
warmup schedule.

### What it solves and what it does not

| Solves | Does NOT solve |
| ------ | -------------- |
| Throughput (more tokens/sec) | Memory (each GPU still holds full model) |
| Linear data throughput scaling (ideal) | Models too large to fit on one GPU |

### Scaling behavior

In theory, doubling the number of GPUs halves the time per epoch (strong
scaling). In practice, communication overhead means efficiency degrades:

- Within a node (NVLink): near-linear scaling to 8 GPUs
- Across nodes (InfiniBand): efficiency drops; depends on interconnect
  bandwidth vs. model size

### MPI analogy

| MPI concept | DDP concept |
| ----------- | ----------- |
| `MPI_COMM_WORLD` | default process group |
| rank | GPU process index |
| `MPI_Allreduce` | gradient all-reduce |

---

## 2. Tensor Parallelism (TP)

### The idea

For a model too large to fit on one GPU, tensor parallelism splits the
**computation inside a single layer** across multiple GPUs. Each GPU holds
a shard of the layer's weight matrix and computes a partial result. The
partial results are combined with collectives.

### Column parallelism

Split a weight matrix by **output features** (columns):

```
Full weight W: shape [out, in]

GPU 0: W0 = W[0     : out/2, :]  -->  partial output Y0
GPU 1: W1 = W[out/2 : out,   :]  -->  partial output Y1

Combine:  all-gather(Y0, Y1)  -->  full output Y = [Y0 | Y1]
```

Communication: **all-gather** of partial outputs.

### Row parallelism

Split a weight matrix by **input features** (rows):

```
Full input X: shape [batch, in]
Full weight W: shape [out, in]

GPU 0: X0 = X[:, 0     : in/2],  W0 = W[:, 0     : in/2]  -->  partial sum S0
GPU 1: X1 = X[:, in/2 : in  ],  W1 = W[:, in/2 : in  ]  -->  partial sum S1

Combine:  all-reduce(S0 + S1)  -->  full output Y = S0 + S1
```

Communication: **all-reduce** of partial sums.

### Why column then row?

In a transformer, the attention projection and feed-forward network (FFN) are
often implemented as column-parallel followed by row-parallel:

- Column-parallel: first linear, output partitioned across GPUs
- No communication in between (each GPU continues with its own shard)
- Row-parallel: second linear, all-reduce at the end

This **Megatron-LM** approach (Shoeybi et al., 2019) uses exactly two
collectives per transformer layer -- one all-gather and one all-reduce --
and is designed so the collectives happen at layer boundaries, not inside.

### Attention head splitting

Multi-head attention is naturally tensor-parallel:

- `H` heads split across `p` GPUs: each GPU runs `H/p` heads
- No communication needed during attention computation
- One all-reduce at the output projection

### Cost and constraints

Tensor parallelism requires **low-latency, high-bandwidth** interconnect because
a collective is needed on every forward and backward pass through the split
layers. It is almost always kept within a single node (NVLink connects 8 A100s
at 600 GB/s bidirectional).

Across slower interconnects (InfiniBand, Ethernet), the communication overhead
negates the benefit.

---

## 3. Pipeline Parallelism (PP)

### The idea

Divide the model's **layers** across different GPUs or nodes. Each device is
a "stage" in a pipeline. Data flows through the stages like an assembly line.

```
Stage 0 (GPU 0):  layers  1 -- 12   (embed + first 12 transformer blocks)
Stage 1 (GPU 1):  layers 13 -- 24
Stage 2 (GPU 2):  layers 25 -- 36
Stage 3 (GPU 3):  layers 37 -- 48 + LM head
```

### Micro-batches

To keep all stages busy, the global batch is split into **micro-batches** that
flow through the pipeline one at a time:

```
Time  -->

Stage 0:  [mb1 fwd]  [mb2 fwd]  [mb3 fwd]  [mb4 fwd]  .  .  [mb4 bwd]  [mb3 bwd]  ...
Stage 1:  .  [mb1 fwd]  [mb2 fwd]  [mb3 fwd]  [mb4 fwd]  .  .  [mb4 bwd]  ...
Stage 2:  .  .  [mb1 fwd]  ...
Stage 3:  .  .  .  [mb1 fwd+bwd]  ...
```

### Pipeline bubble

Stages are idle during startup (all stages must receive data before computing)
and drain (the final stage finishes before earlier stages finish their backward
pass). This idle time is called the **pipeline bubble**.

Bubble fraction ≈ `(p - 1) / (m + p - 1)` where `p` is the number of stages
and `m` is the number of micro-batches.

For `p = 4` stages and `m = 8` micro-batches: bubble ≈ 3/11 ≈ 27%.
For `m = 32`: bubble ≈ 3/35 ≈ 9%.

More micro-batches reduce the bubble but increase memory (more activations
in flight).

### 1F1B schedule

The **1 Forward 1 Backward** schedule (Narayanan et al., 2021) interleaves
forward and backward micro-batches to reduce pipeline bubble:

```
Stage 0:  [1F]  [2F]  [3F]  [4F]  [4B]  [3B]  [2B]  [1B]
Stage 1:   .   [1F]  [2F]  [3F]  [3B]  [2B]  [1B]  ...
```

This is the schedule used by Megatron-LM.

### Communication

Pipeline stages communicate via **point-to-point send/recv**: stage `k`
sends its output activation to stage `k+1` in the forward pass, and sends
the gradient back to stage `k-1` in the backward pass.

This is the same as `MPI_Send` / `MPI_Recv` between neighboring ranks.

### What it solves

| Solves | Cost |
| ------ | ---- |
| Model too deep to fit one GPU | Pipeline bubble (idle GPU time) |
| Reduces memory per GPU | More complex scheduling and debugging |

---

## 4. ZeRO and FSDP: Sharded Model State

### The problem with data parallelism memory

In plain DDP, every GPU stores:
- Full parameters (16 bytes/param in mixed precision)
- Full gradients (2 bytes/param fp16)
- Full optimizer state (8 bytes/param fp32 for Adam)

If you have 8 GPUs in a data-parallel group, each GPU stores the entire 16
bytes/param -- there is no saving from the 8 replicas.

### ZeRO: Zero Redundancy Optimizer (Rajbhandari et al., 2020)

ZeRO eliminates this redundancy by sharding model state across the data-parallel
ranks. There are three levels:

#### ZeRO Stage 1 -- shard optimizer state

Each GPU holds the full parameters and gradients, but only `1/N` of the
optimizer states (first and second Adam moments, master weights).

Memory saving: eliminates 8 bytes/param × (N-1)/N ≈ 75% of optimizer state
for large N.

#### ZeRO Stage 2 -- shard gradients too

Each GPU holds the full parameters, but only `1/N` of the gradients and
optimizer states.

After each backward pass:
1. Reduce-scatter: each GPU keeps only its shard of the gradients
2. Optimizer step: each GPU updates only its shard of the parameters
3. All-gather: reconstruct full parameters for the next forward pass

Memory saving: eliminates ~87% of per-GPU redundancy (gradients + optimizer
state).

#### ZeRO Stage 3 -- shard parameters too

All parameters, gradients, and optimizer states are sharded.

Before each forward pass through a layer, the participating GPUs do an
**all-gather** to reconstruct the layer parameters. After use, the non-owner
shards are discarded.

Memory saving: eliminates all redundancy. Memory per GPU scales as
`total_model_state / N`.

#### Memory comparison (N = 64 GPUs, 175B params)

| Strategy | Memory per GPU |
| -------- | -------------- |
| DDP (baseline) | 2.8 TB |
| ZeRO Stage 1 | ~0.8 TB |
| ZeRO Stage 2 | ~0.35 TB |
| ZeRO Stage 3 | ~44 GB ✓ |

ZeRO Stage 3 makes GPT-3-scale training possible on a cluster of standard
GPUs.

### FSDP: Fully Sharded Data Parallel

**Fully Sharded Data Parallel** is PyTorch's built-in implementation of ZeRO
Stage 3, added in PyTorch 1.11. It wraps each module individually:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)
```

Key differences from DDP:
- Parameters are sharded across GPUs at rest
- Before a module's forward, parameters are all-gathered
- After a module's forward/backward, the non-owner shards are freed
- Each GPU only stores `1/world_size` of every parameter

### Communication pattern for ZeRO / FSDP

| Operation | Collective |
| --------- | ---------- |
| Reconstruct full params for forward | all-gather |
| Accumulate gradients | reduce-scatter |
| Optimizer step | local (on each shard) |

The total communication volume is the same as plain DDP all-reduce
(each parameter crosses the network once), but the memory footprint is
drastically reduced.

### ZeRO-Infinity

An extension that spills optimizer states to CPU RAM or NVMe storage when GPU
memory is exhausted. Enables training models far larger than the total GPU
memory pool, at the cost of CPU-GPU transfer overhead.

---

## 5. Combining Strategies: 3D Parallelism

Real production training for very large models uses **all four strategies
simultaneously**:

```
World = (D × T × P) GPUs

D  data-parallel groups     -- handle different token batches
T  tensor-parallel groups   -- split layers within a node (NVLink)
P  pipeline stages          -- split model depth across nodes
```

Example: training a 530B-parameter model on 2048 A100 GPUs

```
Pipeline stages:  P = 8   (each stage holds ~66B params across 256 GPUs)
Tensor parallel:  T = 8   (within a node; 8 A100s per node)
Data parallel:    D = 32  (32 independent model replicas)

Total: 8 × 8 × 32 = 2048 GPUs
```

Each of the 8 pipeline stages contains 32 × 8 = 256 GPUs. Within each stage,
8 GPUs share the layer via tensor parallelism, and 32 such groups run on
different data shards in data parallelism.

This is the approach used in Megatron-LM for the 530B Megatron-Turing NLG
model (Smith et al., 2022) and in similar production systems at Google,
Meta, and others.

---

## 6. Decision Guide: Which Strategy to Use

| Situation | Recommended strategy |
| --------- | -------------------- |
| Model fits on one GPU | DDP only |
| Model fits with ZeRO Stage 3 | FSDP (ZeRO Stage 3) + DDP |
| Model needs multiple nodes | Pipeline parallelism + tensor parallelism + DDP |
| Very long sequences (context > 32K) | Sequence parallelism (extension of TP) |
| Want to minimize code changes | FSDP (PyTorch built-in) |
| Maximum throughput, HPC cluster | Megatron-LM (3D parallelism) |

---

## 7. Communication Summary

| Strategy | Collective | Where | Bandwidth requirement |
| -------- | ---------- | ----- | --------------------- |
| Data parallelism | all-reduce | across all DP ranks | medium (once per step) |
| Tensor parallelism | all-gather + all-reduce | within TP group | very high (every layer) |
| Pipeline parallelism | send/recv | between adjacent stages | medium (activation size) |
| ZeRO Stage 2 | reduce-scatter | within DP group | medium (gradient size) |
| ZeRO Stage 3 | all-gather + reduce-scatter | within DP group | high (every module) |

---

## 8. Exercises

1. A transformer has 96 layers and you have 8 pipeline stages. How many layers
   per stage? What happens if layer compute times are unequal?

2. You have 4 GPUs and want tensor parallelism with 4-way splits on attention
   heads. The model has 32 attention heads. How many heads does each GPU
   handle?

3. With ZeRO Stage 3 and 64 GPUs, how much memory does each GPU need for a
   70B-parameter model (using 16 bytes/param for model state)?

4. Why is tensor parallelism restricted to within-node groups but pipeline
   parallelism can span across nodes?

5. In 3D parallelism with `T=4, P=4, D=8`, how many total GPUs are needed?
   If you double D only, what changes about memory and throughput?
