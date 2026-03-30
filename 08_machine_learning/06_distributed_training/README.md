# 06 Distributed Training: Data Parallelism

This lesson introduces distributed PyTorch training through data parallelism.

The recommended approach for this course is PyTorch Distributed Data Parallel
(DDP) rather than Horovod. That keeps the software stack smaller and matches the
rest of the course better.

Topics to cover:

- replicated model on each process
- partitioned data across processes
- gradient synchronization
- rank, world size, and local device selection

Learning goals:

- connect data parallelism to MPI-style SPMD thinking
- understand the role of collective communication in training
- measure strong scaling on a familiar workload

Suggested implementation path:

1. start from the single-node MNIST example
2. launch multiple processes
3. shard the dataset by rank
4. synchronize gradients each step
5. report timing on rank 0

Suggested run:

```bash
torchrun --nproc_per_node=2 main.py --epochs 3 --batch-size 128
```

Suggested setup:

```bash
python3 -m pip install torch
```

Notes:

- begin with a single node before discussing multi-node launches
- keep checkpointing and logging simple in the first version
