# Exercise: Tensor-Parallel Matrix Multiplication

This exercise demonstrates tensor parallelism on one matrix multiplication:

```text
C = A @ B
```

The script splits the shared `K` dimension across ranks. Each rank computes a
partial matrix product, then all ranks use `all_reduce(SUM)` to reconstruct the
full result.

## Files

| File | Description |
|---|---|
| `matmul_tensor_parallel.py` | K-sharded matrix multiplication demo using `torch.distributed`. |

## Parallel Pattern

For `world_size = P`, the `K` dimension is partitioned into `P` shards:

```text
A : [M, K]      B : [K, N]

rank i owns:
  A_i = A[:, k_start:k_end]
  B_i = B[k_start:k_end, :]

local_C_i = A_i @ B_i
C = sum_i local_C_i
```

Because each `local_C_i` has shape `[M, N]`, the reconstruction step is an
all-reduce sum, not a gather.

## Run

Single process:

```bash
python3 matmul_tensor_parallel.py
```

Two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 matmul_tensor_parallel.py
```

Four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 matmul_tensor_parallel.py
```

On Polaris, launch through MPI and the local exercise launcher from another
exercise directory if needed, or use `torchrun` inside an allocated interactive
GPU job.

## Output

Rank 0 prints:

- matrix shapes
- world size
- local `K` slice size
- local matmul time
- all-reduce time
- maximum absolute and relative error against a reference `A @ B`

The script ends with `PASSED` if the tensor-parallel result matches the
reference within tolerance.

## Constraints

- `K` must be divisible by `world_size`.
- The default matrices are large: `M = K = N = 16384`.
- Every rank currently constructs full `A` and `B` so the script can check
  correctness. A production implementation would store only local shards.

If the default problem size is too large for the available CPU or GPU memory,
reduce `M`, `K`, and `N` near the top of `main()`.

## Questions

1. Which dimension is sharded across ranks?
2. Why does this implementation use `all_reduce(SUM)`?
3. What would change if `B` were split by columns instead?
4. How do local matmul time and all-reduce time change as GPU count increases?
5. Why does this demo still allocate the full matrices on every rank?

## Extension

Modify the script to accept `M`, `K`, and `N` as command-line arguments. Then
run a sweep over matrix sizes and GPU counts to determine when communication
cost dominates local compute.
# Request 2 nodes