# 06 Distributed Training: Data Parallelism

This lesson introduces distributed PyTorch training through data parallelism.

The recommended approach for this course is PyTorch Distributed Data Parallel
(DDP) rather than Horovod. That keeps the software stack smaller and matches the
rest of the course better.

Main topics:

- replicated model on each process
- partitioned data across processes
- gradient synchronization
- rank, world size, and local device selection

Learning goals:

- connect data parallelism to MPI-style SPMD thinking
- understand the role of collective communication in training
- measure strong scaling on a familiar workload

Suggested teaching path:

1. start from the single-node MNIST example
2. launch multiple processes
3. shard the dataset by rank
4. synchronize gradients each step
5. report timing on rank 0

Files in this directory:

- `train_mnist_ddp.py`
  MNIST CNN example for DDP scaling studies.
- `train_cifar10_ddp.py`
  CIFAR-10 CNN example for DDP scaling studies.
- `run_scaling_qsub.sh`
  Polaris PBS script that allocates 2 nodes and runs `mpiexec` cases for
  `1`, `2`, `4`, and `8` GPUs.
- `DP.md`
  Longer lecture note explaining DDP, scaling, and warmup ideas.
- `cifar10_example_python/`
  A lecture-ready CIFAR-10 PyTorch DDP example using `DistributedSampler`,
  a small CNN, and global metric reduction.
- `DATA_PARALLEL_TRAINING.md`
  Short teaching note with schematic plots for replicated models, sharded data,
  and gradient synchronization.

Recommended software:

- Python 3
- PyTorch
- torchvision

Suggested setup:

```bash
python3 -m pip install torch torchvision
```

Single-node examples:

```bash
torchrun --nproc_per_node=2 train_mnist_ddp.py --epochs 5 --batch-size 64
torchrun --nproc_per_node=2 train_cifar10_ddp.py --epochs 10 --batch-size 128
```

Polaris scaling run:

```bash
qsub run_scaling_qsub.sh
```

The PBS script reserves `2` Polaris nodes and runs repeated `mpiexec` launches
for `1`, `2`, `4`, and `8` GPUs. Results are appended to
`results/scaling_results.csv`.

Teaching notes:

- begin with a single node before discussing multi-node launches
- keep checkpointing and logging simple in the first version
- use `DP.md` when you want the longer walkthrough
- use `DATA_PARALLEL_TRAINING.md` when you want a shorter schematic-first note

## Measured Scaling Results

The following measurements were collected from the MNIST and CIFAR-10 DDP
examples using `1`, `2`, `4`, and `8` GPUs.

### Raw Results

| Model | GPUs | Batch/GPU | Global Batch | Base LR | Scaled LR | Avg Epoch Time (s) | Throughput (samples/s) | Total Time (s) | Final Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | 1 | 64 | 64 | 0.001 | 0.001 | 2.343 | 25604.1 | 14.147 | 99.3 |
| MNIST | 2 | 64 | 128 | 0.001 | 0.002 | 1.516 | 39566.8 | 10.079 | 98.9 |
| MNIST | 4 | 64 | 256 | 0.001 | 0.004 | 1.014 | 59185.3 | 7.887 | 98.8 |
| MNIST | 8 | 64 | 512 | 0.001 | 0.008 | 0.870 | 69004.1 | 6.762 | 98.6 |
| CIFAR-10 | 1 | 128 | 128 | 0.1 | 0.1 | 5.264 | 9498.4 | 58.491 | 78.6 |
| CIFAR-10 | 2 | 128 | 256 | 0.1 | 0.2 | 2.794 | 17896.0 | 33.836 | 79.8 |
| CIFAR-10 | 4 | 128 | 512 | 0.1 | 0.4 | 1.552 | 32224.1 | 21.600 | 69.7 |
| CIFAR-10 | 8 | 128 | 1024 | 0.1 | 0.8 | 1.029 | 48571.5 | 16.137 | 43.2 |

### Speedup Bar Plot

The plot below uses throughput-based speedup relative to the 1-GPU baseline for
each model.

![Throughput-based speedup for MNIST and CIFAR-10 DDP runs](assets/scaling_speedup_bar_plot.svg)

### Observations

- MNIST shows modest scaling because the model is small and communication
  overhead becomes noticeable quickly.
- CIFAR-10 shows stronger throughput scaling because there is more computation
  per step to amortize communication.
- CIFAR-10 accuracy drops sharply at larger GPU counts because the experiment is
  changing both the global batch size and the scaled learning rate.
- These results are therefore useful for discussing both parallel scaling and
  the difference between system performance and optimization behavior.
