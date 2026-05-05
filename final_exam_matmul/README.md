# Final Exam — CUDA Matrix-Matrix Multiplication

CSCI 394 High Performance Computing | Spring 2026

## Overview

Complete the CUDA kernel and host code in `matmul.cu` to compute **C = A × B**
for square FP32 matrices of sizes n = 16, 64, 256, 1024, and 4096.
The program measures the average kernel time over 10 iterations and reports
achieved GFLOPS and hardware efficiency relative to the A100's peak (19,500 GFLOPS).

## Files

| File | Description |
|------|-------------|
| `matmul.cu` | Starter code — complete all `TODO` sections |
| `Makefile` | Build with `make` |
| `qsub_polaris.sh` | PBS job script for Polaris (reservation queue) |

## Your Tasks

Open `matmul.cu` and complete the six `TODO` steps:

1. **Kernel inner loop** — accumulate `C[row][col] = Σ_k A[row][k] * B[k][col]`
2. **Allocate** device memory (`cudaMalloc`) for `d_A`, `d_B`, `d_C`
3. **Copy** `h_A` and `h_B` to the device (`cudaMemcpy`)
4. **Set grid dimensions** — `dim3 grid((n+BLK-1)/BLK, (n+BLK-1)/BLK)`
5. **Compute** `gflops` and `efficiency` from the measured time
6. **Copy** `d_C` back to `h_C` and **free** all device memory

## Build

```bash
module load nvhpc/23.3
make
```

## Run on Polaris (reservation queue)

A PBS reservation is available on **Tue May 5, 2026, 8:00–11:00 AM CT**.

```bash
qsub qsub_polaris.sh
```

Or interactively:

```bash
qsub -I -l select=1:system=polaris \
        -l walltime=00:30:00       \
        -l filesystems=home:eagle  \
        -q R7115122 -A DLIO

cd $PBS_O_WORKDIR
module load nvhpc/23.3
make
./matmul
```

## Expected Output Format

```
n       time (ms)     GFLOPS          efficiency
-------  ------------  --------------  ----------
n=   16  time=   x.xxx ms  GFLOPS=     x.x  efficiency=  x.x%
n=   64  ...
n=  256  ...
n= 1024  ...
n= 4096  ...
```

## Questions to Answer (on exam sheet)

- **(a)** Write the inner loop body for the kernel.
- **(b)** Write the expressions for `gflops` and `efficiency`.
- **(c)** Record your measured results in the table on the exam sheet.
- **Discussion:** Why does efficiency increase with n? What optimization
  would improve efficiency at all sizes?
