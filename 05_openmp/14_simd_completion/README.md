# OpenMP study: SIMD and thread-level parallelism

This module demonstrates the interplay between SIMD vectorization and OpenMP thread parallelism with two kernels:
- `axpy_simd`: AXPY (`out[i] = a*x[i] + y[i]`)
- `matmul_simd`: dense matrix multiplication (`C = A x B`, `N x N`)

## What is compared
- `serial` (no OpenMP pragma)
- `omp simd`
- `omp parallel for`
- `omp parallel for simd`

Each program reports the same four cases:
- best time across repeats,
- mean time across repeats,
- checksum (for sanity check).

## Build
```bash
make
```

## Run
```bash
# AXPY version
./axpy_simd
OMP_NUM_THREADS=4 ./axpy_simd 20000000 10

# MatMul version
./matmul_simd
OMP_NUM_THREADS=4 ./matmul_simd 1024 3
```

## Requested experiments
```bash
# AXPY experiment
OMP_NUM_THREADS=2 ./axpy_simd 1638400 10

# MatMul experiment
OMP_NUM_THREADS=2 ./matmul_simd 512 3
```

## Notes
- Thread count is controlled by `OMP_NUM_THREADS`.
- `axpy_simd`: `N` is vector length, default repeats is `10`.
- `matmul_simd`: `N` is matrix dimension (`O(N^3)`), default repeats is `3`.
