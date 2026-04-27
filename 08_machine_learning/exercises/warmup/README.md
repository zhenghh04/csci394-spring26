# Exercise: DDP Learning-Rate Warmup on MNIST

This exercise studies why learning-rate warmup is often needed when data
parallel training increases the effective batch size.

The script trains a small MNIST CNN with PyTorch Distributed Data Parallel
(DDP). Each GPU runs one process, receives a shard of the training data, and
uses DDP to all-reduce gradients after the backward pass.

## Files

| File | Description |
|---|---|
| `train_mnist_ddp.py` | MNIST DDP training script with linear LR scaling and optional warmup. |
| `launcher.sh` | Polaris/PBS helper that sets rank and world-size environment variables before launching Python. |

## What To Study

When the number of GPUs increases, the per-step effective batch size is:

```text
effective_batch_size = batch_size_per_gpu * world_size
```

The script applies the linear scaling rule:

```text
scaled_lr = base_lr * world_size
```

With warmup enabled, the learning rate ramps from `base_lr` to `scaled_lr`
over the first `warmup_epochs` epochs.

## Run Locally With torchrun

Use `torchrun` even for a one-GPU run because the script initializes a
distributed process group.

```bash
torchrun --standalone --nproc_per_node=1 train_mnist_ddp.py \
  --epochs 5 --batch-size 64 --lr 0.01 --warmup-epochs 0
```

On a multi-GPU node:

```bash
torchrun --standalone --nproc_per_node=4 train_mnist_ddp.py \
  --epochs 5 --batch-size 64 --lr 0.01 --warmup-epochs 0

torchrun --standalone --nproc_per_node=4 train_mnist_ddp.py \
  --epochs 5 --batch-size 64 --lr 0.01 --warmup-epochs 2
```

## Run On Polaris

From this directory, use the launcher so each MPI rank receives the environment
variables expected by PyTorch distributed:

```bash
mpiexec -np 4 --ppn 4 --cpu-bind depth -d 16 \
  ./launcher.sh python3 train_mnist_ddp.py \
  --epochs 5 --batch-size 64 --lr 0.01 --warmup-epochs 2
```

Adjust `-np` and `--ppn` to match the number of GPUs requested in the PBS job.

## Output

Rank 0 prints the training summary and appends one row to:

```text
results/scaling_results.csv
```

The CSV records GPU count, batch size, base learning rate, scaled learning
rate, average epoch time, throughput, total runtime, and final accuracy.

## Suggested Experiments

Run at least these cases:

| GPUs | Warmup epochs | Purpose |
|---:|---:|---|
| 1 | 0 | Single-GPU baseline. |
| 2 | 0 | Data-parallel run without warmup. |
| 4 | 0 | Larger effective batch without warmup. |
| 4 | 2 | Same scale with warmup. |
| 4 | 5 | Longer warmup comparison. |

## Questions

1. How does throughput change as the number of GPUs increases?
2. How does final accuracy change without warmup?
3. Does warmup recover accuracy at larger GPU counts?
4. What is the effective batch size in each run?
5. Why does DDP require `DistributedSampler` for the training dataset?

## Notes

- The first run downloads MNIST into `data/`.
- The script uses the NCCL backend and expects CUDA GPUs.
- If you cannot access multiple GPUs, run the one-GPU baseline and explain
  which distributed cases you could not test.
