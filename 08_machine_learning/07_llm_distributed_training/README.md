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
| `train_transformer_3d_parallel.py` | **3D parallelism benchmark**: TP + PP + DP on an 8-layer transformer |
| `run_3d_parallel_scaling.sh` | PBS script: sweeps all valid (tp, pp) combos on 2 Polaris nodes |

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

## 3D Parallelism Scaling Study (Polaris)

`train_transformer_3d_parallel.py` implements a decoder-only transformer that
combines all three parallelism strategies simultaneously:

- **Tensor Parallelism (TP)**: attention heads and FFN output features are split
  across `tp_size` GPUs using Megatron-LM style column/row parallel linear layers.
  Within-node TP uses fast NVLink; cross-node TP is slow.
- **Pipeline Parallelism (PP)**: the 8 transformer layers are divided into
  `pp_size` stages. Activations flow forward stage-by-stage; gradients flow back
  via blocking NCCL send/recv (GPipe-style sequential schedule).
- **Data Parallelism (DP)**: the entire TP×PP mesh is replicated `dp_size` times.
  Gradients are manually all-reduced across DP replicas after each backward pass.

Global rank layout: `rank = dp * pp_size * tp_size + pp * tp_size + tp`

### Running on 2 Polaris nodes (8 × A100 GPUs total)

```bash
qsub run_3d_parallel_scaling.sh
```

The PBS script sweeps 10 configurations in a single 45-minute job:

| TP | PP | DP | Notes |
|----|----|----|-------|
| 1  | 1  | 8  | Pure data parallelism (baseline) |
| 2  | 1  | 4  | TP=2 within each node |
| 4  | 1  | 2  | TP=4 fills one node |
| 8  | 1  | 1  | TP spans both nodes (crosses slow IB — expected slow) |
| 1  | 2  | 4  | PP only, one stage per node |
| 2  | 2  | 2  | TP within node + PP across nodes — expected winner |
| 4  | 2  | 1  | Strong TP + PP across nodes — potential winner |
| 1  | 4  | 2  | Deeper pipeline, 2 stages per node |
| 2  | 4  | 1  | TP + deeper pipeline |
| 1  | 8  | 1  | Maximum pipeline depth, 1 layer/stage (high bubble) |

Results are written to `results_3d_parallel/scaling_results.csv` and printed as
a sorted summary table at the end of the job.

### Expected outcome

Configurations that keep TP within a single node (tp ≤ 4 on Polaris) avoid
slow inter-node NVLink and should outperform `tp=8`. The best throughput is
typically `tp=4, pp=2` or `tp=2, pp=2` — TP uses fast intra-node NVLink while
PP crosses nodes only once per layer group with a single activation tensor.

### Running a single config manually

```bash
# Example: TP=2, PP=2, DP=2 on 8 GPUs
mpiexec -n 8 --ppn 4 --hostfile $PBS_NODEFILE \
  --env MASTER_ADDR=<node0> --env MASTER_PORT=29500 \
  bash -lc "
    export RANK=$PMI_RANK LOCAL_RANK=$PMI_LOCAL_RANK WORLD_SIZE=8
    python train_transformer_3d_parallel.py \
      --tp-size 2 --pp-size 2 \
      --num-layers 8 --d-model 512 --num-heads 8 --ff-dim 2048 \
      --seq-len 256 --batch-size 4 --num-iters 20 --warmup-iters 5
  "
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

6. **3D parallelism** (`LLM_DISTRIBUTED_TRAINING.md` §6, `train_transformer_3d_parallel.py`)
   How the production stacks compose all strategies simultaneously. Run the
   scaling study and ask students to predict which config wins before seeing results.

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
