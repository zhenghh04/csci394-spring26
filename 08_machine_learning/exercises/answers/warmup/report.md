# Warmup Exercise — Answers

Source: [`../../warmup/train_mnist_ddp.py`](../../warmup/train_mnist_ddp.py)

The script trains a small MNIST CNN with PyTorch DDP. Per-rank batch size is
fixed; the effective batch grows with `world_size`; `lr` is linearly scaled
and (optionally) warmed up over the first `warmup_epochs` epochs.

Key code references:
- linear scaling: `train_mnist_ddp.py:147` (`scaled_lr = args.lr * world_size`)
- warmup ramp:    `train_mnist_ddp.py:177-182`
- DistributedSampler shuffle reset: `train_mnist_ddp.py:184` (`set_epoch`)

## Recommended sweep

Run from `../../warmup/`:

```bash
# 1 GPU baseline
mpiexec -np 1 --ppn 1 --cpu-bind depth -d 16 \
  ./launcher.sh python3 train_mnist_ddp.py --epochs 5 --batch-size 64 --lr 0.01 --warmup-epochs 0

# 2 GPU, no warmup
mpiexec -np 2 --ppn 2 ... --warmup-epochs 0

# 4 GPU sweep
mpiexec -np 4 --ppn 4 ... --warmup-epochs 0
mpiexec -np 4 --ppn 4 ... --warmup-epochs 2
mpiexec -np 4 --ppn 4 ... --warmup-epochs 5
```

The included `run_polaris.sh` runs the full table.
Each row is appended to `results/scaling_results.csv`.

---

## Q1. How does throughput change as the number of GPUs increases?

Throughput (samples/sec) grows roughly linearly with `world_size`, but with
sub-linear efficiency for small models like this MNIST CNN.

Reason: each rank still processes the same `batch_size` per step (here 64),
so per-step compute is constant — only the global samples/step grows. The
fixed cost per step is dominated by:

- the gradient all-reduce that DDP installs as a backward hook (scales with
  the model size, not the batch);
- DataLoader/host-to-device transfer overhead;
- per-step CUDA launch latency.

For a 1.2 M-param model on a 60 k-sample dataset, the all-reduce is cheap
relative to compute, so scaling efficiency is typically 0.85–0.95 from 1→4
GPUs on Polaris (NVLink-connected A100s). On 1 node beyond ~4 ranks the
DataLoader workers (`num_workers=4` per rank → 16 total) start to contend
for the host CPU and efficiency drops.

What to look for in the CSV: `avg_throughput_samples_per_sec` should grow
~3–4× from 1 to 4 GPU; total wall time should drop ~3–3.5×.

## Q2. How does final accuracy change without warmup?

Without warmup, accuracy is essentially unchanged at 1→2 GPUs and starts to
visibly degrade at 4 GPUs. Concretely, after 5 epochs you should see:

| GPUs | warmup | typical final acc |
|---:|---:|---|
| 1 | 0 | ~99.0–99.2% |
| 2 | 0 | ~98.9–99.1% |
| 4 | 0 | ~98.5–98.9% (wider variance) |
| 4 | 2 | back to ~99.1% |
| 4 | 5 | ~99.1% (no further gain) |

Why: the linear scaling rule `lr ← world_size · lr` is correct **on average**
but not at step 1, when the model is randomly initialized. A 4× larger
learning rate applied to noisy initial gradients pushes the optimizer into a
worse basin. The effect on MNIST is mild (the loss landscape is forgiving)
but reproducible. On harder datasets (CIFAR, ImageNet) the same recipe
without warmup can lose several percent or fail to converge.

## Q3. Does warmup recover accuracy at larger GPU counts?

Yes. With `--warmup-epochs 2` at 4 GPUs, accuracy returns to within noise of
the 1-GPU baseline. Going from 2 → 5 warmup epochs gives no further benefit
on this problem because the model is already near its asymptote by epoch 3;
extra warmup just spends those epochs at a sub-optimal LR. As a rule of
thumb, warmup length is an LR concern, not a model concern: it should be
long enough for the parameters to leave the random-init regime and short
enough that you spend most of your epochs at the scaled LR.

## Q4. What is the effective batch size in each run?

`effective_batch_size = batch_size_per_gpu × world_size` (`train_mnist_ddp.py:107`).

| GPUs | per-GPU batch | effective batch |
|---:|---:|---:|
| 1 | 64 | 64 |
| 2 | 64 | 128 |
| 4 | 64 | 256 |

DDP does not change the per-GPU batch; instead it changes how many
mini-batches are processed in parallel before the optimizer step. The
optimizer therefore sees an aggregated gradient computed from
`effective_batch_size` samples and takes one step per such aggregate.

## Q5. Why does DDP require `DistributedSampler` for the training dataset?

Three reasons, in order of importance:

1. **Non-overlapping shards.** Without `DistributedSampler`, every rank
   would iterate the full dataset, so every sample would be visited
   `world_size` times per epoch and gradient averaging would cancel
   nothing — you would do `world_size`× the work for the same effective
   data per step.
2. **Synchronized epoch length.** All ranks must take the same number of
   optimizer steps so that the all-reduce after `loss.backward()` finds a
   matching call on every other rank. `DistributedSampler` rounds the
   per-rank length to the same value (padding by re-using a few samples
   when `len(dataset) % world_size != 0`).
3. **Coherent shuffling.** `set_epoch(epoch)` (`train_mnist_ddp.py:184`)
   seeds the sampler with a value all ranks agree on, so each epoch sees a
   fresh permutation of the full dataset partitioned identically across
   ranks. Skipping `set_epoch` would re-use the same per-rank shard order
   every epoch — equivalent to disabling shuffling.

Note that `DistributedSampler` is only needed for the training set. The
test set (`train_mnist_ddp.py:135`) is iterated unsharded because each
rank already holds the full model and we just want a quick accuracy check
on rank 0.

## Notes on the script

- `dist.init_process_group(..., device_id=device)` (line 98) is the
  newer per-device-aware init API. It avoids the legacy implicit
  `cuda:LOCAL_RANK` binding.
- The "warmup iteration" before the timed loop (line 162) is unrelated to
  the LR warmup; it touches NCCL and CUDA caches so the first timed epoch
  is not biased by one-time init costs.
- `total_samples = len(train_dataset)` is the whole dataset, not the
  per-rank shard, so the reported throughput is global samples/sec.
