# Distributed Training for Large Language Models

This note extends the DDP material in this folder from image-classification
examples to the systems ideas needed for training modern large language models
(LLMs).

Recommended prerequisite:

- read `README.md` and `DATA_PARALLEL_TRAINING.md` first
- use `DP.md` for the full DDP walkthrough before discussing LLM-scale systems

---

## 1. Why LLM Training Is Different

The basic training loop is still the same:

1. forward pass
2. loss computation
3. backward pass
4. optimizer step

What changes is the scale.

For an LLM, we are no longer training a small CNN that fits comfortably on one
GPU. We now care about:

- billions of parameters
- long token sequences
- very large training corpora
- optimizer state memory
- communication cost across many GPUs and nodes

This creates two separate problems:

1. **capacity**
   Can the model, optimizer state, gradients, and activations fit in memory?
2. **throughput**
   Can we train fast enough to finish in days or weeks rather than months?

Data parallelism helps throughput, but for sufficiently large models it does
not solve the memory problem by itself.

---

## 2. Why Plain Data Parallelism Stops Being Enough

In ordinary DDP, every GPU stores a full model replica.

That works well when the model fits on one accelerator:

- each rank has the full model
- each rank processes different data
- gradients are synchronized with an all-reduce

For LLMs, memory becomes the first wall.

Rough rule of thumb for Adam-style training in mixed precision:

- parameters, gradients, and optimizer state together often cost roughly
  `16--20 bytes per parameter`
- activations require additional memory during the forward/backward pass

Example:

- `7B` parameters at `16 bytes/parameter` is about `112 GB`
- that estimate is already larger than a single `80 GB` A100
- the true requirement is higher once activations and temporary buffers are
  included

So even before worrying about speed, a full-replica approach may not fit.

---

## 3. Main Parallelism Strategies

Modern LLM training usually combines several forms of parallelism.

| Strategy | What is split | Main benefit | Main cost |
| --- | --- | --- | --- |
| Data parallelism | training data | higher throughput | gradient communication |
| Tensor parallelism | tensors inside a layer | model fits across multiple GPUs | communication inside each layer |
| Pipeline parallelism | stacks of layers | spreads deep models across devices | pipeline bubbles, scheduling complexity |
| FSDP / ZeRO | parameters, gradients, optimizer state | reduces memory per GPU | more frequent gather/scatter traffic |
| Sequence or context parallelism | sequence dimension | helps with long-context training | attention-related communication |

![LLM parallelism strategies](../tutorials/figures/llm_parallelism.svg)

### 3.1 Data Parallelism

This is the direct continuation of the DDP examples in this folder.

- each GPU keeps a full copy of the model
- each GPU sees different token batches
- gradients are averaged across ranks

This is conceptually the easiest strategy and the closest match to MPI SPMD
thinking.

### 3.2 Tensor Parallelism

Here, a single layer is partitioned across multiple GPUs.

Examples:

- split a large matrix multiply by columns or rows
- split attention heads across GPUs
- combine partial results with all-gather or reduce-scatter

This is usually done within a node or within a tightly connected group because
the communication happens during almost every layer.

Good mental model:

- DDP splits **examples**
- tensor parallelism splits **math inside one layer**

### 3.3 Pipeline Parallelism

Here, different groups of layers live on different GPUs or nodes.

Example:

- stage 0 holds layers `1--12`
- stage 1 holds layers `13--24`
- stage 2 holds layers `25--36`
- stage 3 holds layers `37--48`

Micro-batches flow through the stages like an assembly line.

This reduces memory pressure because no single GPU stores every layer, but it
introduces new scheduling issues:

- pipeline startup and drain time
- idle stages when the pipeline is not full
- more complicated debugging and load balancing

### 3.4 FSDP and ZeRO

These approaches shard the model state instead of fully replicating it.

Instead of every GPU holding everything:

- each rank stores only part of the weights
- each rank stores only part of the gradients
- each rank stores only part of the optimizer state

This is one of the key ideas that makes large-model training practical.

In classroom language:

- DDP says "replicate the model, shard the data"
- FSDP/ZeRO says "shard the model state too"

---

## 4. Communication Patterns to Emphasize

LLM training is an HPC communication problem as much as a machine-learning
problem.

The main collective patterns are:

- **all-reduce**
  used for gradient averaging in data parallel training
- **all-gather**
  used when sharded parameters need to be reassembled for computation
- **reduce-scatter**
  used to combine and redistribute gradients or partial results efficiently
- **broadcast**
  used for initial model state or control data
- **point-to-point send/recv**
  used in pipeline parallel training between neighboring stages

Suggested MPI connection:

| HPC / MPI idea | LLM training analogue |
| --- | --- |
| rank | one training process |
| communicator | process group |
| `MPI_Allreduce` | gradient synchronization |
| `MPI_Reduce_scatter` | sharded gradient/state exchange |
| `MPI_Send` / `MPI_Recv` | pipeline stage transfers |

This helps students see that large-scale AI training is not outside HPC; it is
an application of the same distributed-systems ideas.

---

## 5. A Practical Layered View of LLM Training

When teaching this topic, it helps to separate the system into layers.

### Layer 1: Single-GPU Training

Students should already understand:

- tokens become embeddings
- Transformer blocks perform attention and feed-forward computation
- the training loop computes next-token loss

### Layer 2: DDP Across Multiple GPUs

Start from the examples already in this folder:

- `train_mnist_ddp.py`
- `train_cifar10_ddp.py`

The DDP lesson teaches:

- one process per GPU
- sharded input data
- collective gradient synchronization

### Layer 3: LLM Memory Optimizations

Then explain why LLMs need more:

- mixed precision (`fp16` or `bf16`)
- gradient accumulation
- activation checkpointing
- sharded optimizer state

### Layer 4: Full LLM-Scale Parallelism

Finally, describe the production stack:

- data parallelism across replica groups
- tensor parallelism within fast interconnect groups
- pipeline parallelism for deeper models
- FSDP or ZeRO for memory-efficient model-state sharding

This staged approach keeps the topic from feeling like an unrelated jump in
complexity.

---

## 6. A Simple Worked Example

Suppose we want to train a decoder-only Transformer that does not fit on one
GPU.

One possible decomposition on `64` GPUs is:

- `8` data-parallel groups
- `4`-way tensor parallelism
- `2` pipeline stages

Total GPUs:

`8 x 4 x 2 = 64`

Interpretation:

- each data-parallel group processes different token batches
- within each group, each layer is split across `4` GPUs
- the model depth is split into `2` pipeline stages

This is the kind of multidimensional decomposition used in practice.

The important teaching point is that there is no single "distributed training"
mode for LLMs. Real systems compose several strategies at once.

---

## 7. Performance Questions Worth Discussing

Students often assume that more GPUs automatically means faster training.
That is not guaranteed.

Key bottlenecks:

- **communication overhead**
  gradient exchange and tensor synchronization can dominate runtime
- **memory bandwidth**
  some kernels move data faster than they perform arithmetic
- **pipeline imbalance**
  one slow stage can stall the whole pipeline
- **input pipeline limits**
  tokenization, data loading, and storage can starve accelerators
- **optimization effects**
  larger global batch sizes may require learning-rate changes and warmup

Useful classroom distinction:

- **system scaling**
  how examples or tokens per second change with more GPUs
- **optimization quality**
  whether the model still converges to the same quality

These are related but not identical.

---

## 8. Suggested 20-Minute Teaching Path

1. Start with the DDP idea students already know.
2. Show why full model replication breaks for billion-parameter models.
3. Introduce tensor, pipeline, and sharded training as answers to memory and
   throughput limits.
4. Connect each strategy to a communication pattern students already know from
   MPI.
5. Close with the idea that LLM training is a composition of HPC techniques.

---

## 9. Board-Friendly Summary

If you want a short version to present live:

- DDP alone is good when the model fits on one GPU.
- LLMs often do not fit because parameters, gradients, optimizer states, and
  activations exceed device memory.
- Tensor parallelism splits a layer across GPUs.
- Pipeline parallelism splits model depth across GPUs.
- FSDP/ZeRO shards model state to reduce memory per GPU.
- Real LLM training usually combines several of these strategies.
- The whole topic is fundamentally about balancing compute, memory, and
  communication.

---

## 10. Discussion Prompts

- Why is tensor parallelism usually kept within a node or another
  high-bandwidth group?
- Why can adding GPUs improve throughput but hurt optimization behavior?
- What communication pattern from MPI best matches gradient synchronization?
- Why does a model that fits for inference still fail to fit for training?
- If you had `8` GPUs, would you spend them first on data parallelism or on
  fitting a larger model? Why?

---

## 11. Key Takeaway

Distributed LLM training is not just "DDP on a bigger cluster."

It is a coordinated use of:

- data parallelism for throughput
- tensor and pipeline parallelism for model partitioning
- FSDP/ZeRO for memory efficiency
- collective communication to keep the whole system consistent

That makes LLM training one of the clearest modern examples of HPC ideas in
action.
