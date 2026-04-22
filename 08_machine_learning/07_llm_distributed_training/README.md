# 07 Distributed Training for Large Language Models

This module extends the data parallelism lesson to the systems challenges of
training modern large language models (LLMs). The central theme is that LLM
training is an HPC problem: it demands careful co-design of compute, memory,
and communication.

Prerequisites:
- `05_llm_basics` -- GPT architecture, tokenization, autoregressive decoding
- `06_llm_training` -- single-GPU training loop, loss curves, generation
- `04_distributed_training` -- DDP, collective communication, rank / world size

---

## Why this module exists

A 7B-parameter model trained with Adam requires roughly **112 GB** of memory
just for parameters, gradients, and optimizer states. A single A100 GPU has
80 GB. No single accelerator can hold the model, let alone train it.

Training GPT-3 (175B parameters) on a single A100 would take an estimated
**355 GPU-years**. The actual run used ~10,000 GPUs and finished in weeks.

Distributed training is not an optional performance trick. For LLMs beyond a
certain size it is the only way to train at all.

---

## Topics covered

| File | Topic |
| ---- | ----- |
| `MEMORY_COMPUTE_SCALE.md` | Memory footprint, compute cost, training time estimates |
| `PARALLELISM_STRATEGIES.md` | Data parallelism, tensor parallelism, pipeline parallelism, ZeRO |
| `LLM_DISTRIBUTED_TRAINING.md` | Lecture overview connecting all strategies |
| `train_tiny_llm_ddp.py` | Runnable DDP demo on a tiny LM |
| `tensor_parallel_linear_demo.py` | Column and row parallelism for one linear layer |

---

## Quick start

```bash
# Single process (CPU)
python3 train_tiny_llm_ddp.py --cpu --epochs 10

# Two processes with DDP (CPU, no GPU needed)
torchrun --standalone --nproc_per_node=2 train_tiny_llm_ddp.py --cpu --epochs 10

# Two processes with DDP (GPU)
torchrun --standalone --nproc_per_node=2 train_tiny_llm_ddp.py --epochs 10

# Tensor parallel demo (CPU)
torchrun --standalone --nproc_per_node=2 tensor_parallel_linear_demo.py
torchrun --standalone --nproc_per_node=4 tensor_parallel_linear_demo.py
```

---

## Suggested teaching path

1. **Start from cost** (`MEMORY_COMPUTE_SCALE.md`)
   Show students why distributed training is unavoidable. Memory math is concrete
   and surprising -- most students underestimate it by 10x.

2. **Data parallelism** (`PARALLELISM_STRATEGIES.md` §1)
   Direct continuation of the DDP lesson. Same model replicated, data sharded,
   gradients all-reduced.

3. **Why data parallelism is not enough** (`MEMORY_COMPUTE_SCALE.md` §3)
   When the model does not fit on one GPU, replication is impossible without help.

4. **Tensor and pipeline parallelism** (`PARALLELISM_STRATEGIES.md` §2--3)
   Show the tensor parallel demo. Emphasize the communication pattern.

5. **ZeRO / FSDP** (`PARALLELISM_STRATEGIES.md` §4)
   The modern default for single-host multi-GPU training.

6. **3D parallelism** (`LLM_DISTRIBUTED_TRAINING.md` §6)
   How the production stacks compose all strategies simultaneously.

---

## Key numbers to remember

| Model | Parameters | Training memory (fp16 + Adam) | A100 GPUs needed (min) |
| ----- | ---------- | ----------------------------- | ---------------------- |
| GPT-2 | 1.5B | ~24 GB | 1 (fits) |
| LLaMA-7B | 7B | ~112 GB | 2 |
| LLaMA-13B | 13B | ~208 GB | 3 |
| LLaMA-65B | 65B | ~1 TB | 13 |
| GPT-3 | 175B | ~2.8 TB | 35 |

These numbers are for model state only. Activations, KV cache, and temporary
buffers add more.
