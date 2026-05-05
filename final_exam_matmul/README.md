# Final Exam — CUDA Matrix-Matrix Multiplication

CSCI 394 High Performance Computing | Spring 2026

## Overview

Complete the CUDA kernel and host code in `matmul.cu` to compute **C = A × B**
for square FP32 matrices of sizes n = 16, 64, 256, 1024, 4096, and 16384.
The program measures the average kernel time over 10 iterations (plus one warmup)
and reports achieved GFLOPS and hardware efficiency relative to the A100 peak
(19,500 GFLOPS).

## Files

| File | Description |
|------|-------------|
| `matmul.cu` | Starter code — complete all `TODO` sections |
| `Makefile` | Build with `make` |
| `qsub_polaris.sh` | PBS job script for Polaris (reservation queue) |

## Your Tasks

Open `matmul.cu` and complete the TODO steps:

1. **Kernel inner loop** — accumulate `C[row][col] = Σ_k A[row][k] * B[k][col]`
2. **Allocate** device memory (`cudaMalloc`) for `d_A`, `d_B`, `d_C`
3. **Copy** `h_A` and `h_B` to the device (`cudaMemcpy`)
4. **Set grid dimensions** — `dim3 grid((n+BLK-1)/BLK, (n+BLK-1)/BLK)`
5. **Compute** `gflops` and `efficiency` from the measured time
6. **Copy** `d_C` back to `h_C` and **free** all device memory

> **AI tools are permitted** for this part of the exam. You may use ChatGPT,
> Claude, Copilot, or similar tools to help write and debug the code.
> You must still run the program yourself on Polaris and record your own results.

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
n=16384  ...
```

## Submission — Upload to Canvas

After running on Polaris, **save the terminal output to a text file** and
**upload it to Canvas** under the Final Exam assignment before the end of the
exam period (by **11:00 AM CT, May 5, 2026**).

```bash
# Redirect output to a file
./matmul | tee matmul_results.txt

# Or if using qsub, the output is saved automatically to
# a file named after your job (e.g., DLIO.oXXXXXX)
```

Upload **`matmul_results.txt`** (or the PBS output file) to Canvas.
The grader will verify your results match what you record on the exam sheet.

## Questions to Answer
- **(a)** Record your measured results.
- **(b)** Based on your results, explain why efficiency increases with $n$.
