# OpenMP study: SIMD and thread-level parallelism

This module demonstrates the interplay between SIMD vectorization and OpenMP thread parallelism using three loop forms on the same kernel.

## What is compared
- `omp simd`
- `omp parallel for`
- `omp parallel for simd`

Each case runs the same AXPY-style loop and reports:
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
# N=10000000, repeats=10
./simd_threads_bench

# custom example
OMP_NUM_THREADS=4 ./simd_threads_bench 2000000 10
```

## Requested experiments
```bash
# Experiment 1
OMP_NUM_THREADS=2 ./simd_threads_bench 16384

# Experiment 2
OMP_NUM_THREADS=2 ./simd_threads_bench 1638400
```

## Notes
- Thread count is controlled by `OMP_NUM_THREADS`.
- Problem size is controlled by command-line `N`.
- If `repeats` is not provided, the default is `10`.
