# OpenMP study: SIMD and thread-level parallelism

This module demonstrates the interplay between SIMD vectorization and OpenMP thread parallelism using dense matrix multiplication:
`C = A x B`, where `A`, `B`, and `C` are all `N x N`.

## What is compared
- `omp simd`
- `omp parallel for`
- `omp parallel for simd`

Each case runs the same matrix multiplication kernel and reports:
- best time across repeats,
- mean time across repeats,
- checksum (for sanity check).

## Build
```bash
make
```

## Run
```bash
# Usage:
# ./simd_threads_bench <N> [repeats]

# default values if omitted:
# N=512, repeats=3
./simd_threads_bench

# custom example
OMP_NUM_THREADS=4 ./simd_threads_bench 1024 3
```

## Requested experiments
```bash
# Experiment 1
OMP_NUM_THREADS=2 ./simd_threads_bench 256 3

# Experiment 2
OMP_NUM_THREADS=2 ./simd_threads_bench 1024 3
```

## Notes
- Thread count is controlled by `OMP_NUM_THREADS`.
- Problem size is matrix dimension `N` (work is `O(N^3)`).
- If `repeats` is not provided, the default is `3`.
