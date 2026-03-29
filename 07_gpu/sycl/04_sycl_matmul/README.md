# SYCL Matrix Multiplication

This example computes dense matrix multiplication:

- `C = A * B`

where `A`, `B`, and `C` are square `n x n` matrices.

## Self-explanation

This example extends the earlier vector examples to a two-dimensional dense
matrix kernel.

The program:

1. allocates matrices `A`, `B`, and `C`
2. initializes `A` and `B` on the host
3. launches a 2D SYCL `parallel_for`
4. computes one output entry `C[i, j]` per work-item
5. compares the result with a CPU reference

Why this example matters:

- matrix multiplication is one of the most important kernels in HPC and AI
- it shows how a 2D iteration space maps naturally to a 2D SYCL launch
- it provides a bridge to CUDA GEMM and later Intel XPU assignment work
- it helps students reason about arithmetic work versus memory access

How to read the code:

- `parallel_for(range<2>(n, n), ...)`
  - launches one work-item for each matrix entry
- inside the kernel
  - each work-item computes one dot product for `C[i, j]`
- after `q.wait()`
  - the code computes the CPU reference and reports `max_abs_err`

This is a simple teaching implementation, not a tiled or optimized GEMM.

## Concepts

- 2D `parallel_for`
- dense matrix indexing in row-major storage
- correctness check against a CPU reference
- simple timing with queue synchronization

## Build

```bash
make
```

## Run

```bash
./app
./app 512
```

Arguments:

1. matrix size `n`
