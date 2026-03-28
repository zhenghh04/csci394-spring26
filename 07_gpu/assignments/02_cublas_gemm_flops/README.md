# GPU Assignment: Measure GEMM FLOPs with cuBLAS

## Objective

Use NVIDIA cuBLAS to measure floating-point performance for dense matrix
multiplication on a GPU.

The central question is:

- how does measured GEMM throughput change as matrix size changes?

This assignment also asks you to use an AI coding assistant carefully:

- use ChatGPT to generate an initial CUDA/cuBLAS GEMM program
- then inspect, correct, compile, run, and analyze that code yourself

## Required Program Features

Create your own program in this assignment folder.

Minimum requirement:

1. Use `cublasSgemm` or `cublasDgemm` for dense matrix multiplication.
2. Accept matrix size from the command line.
3. Allocate matrices on the host and device.
4. Initialize input matrices with a reproducible pattern.
5. Time the GPU GEMM call.
6. Report the achieved FLOP/s or GFLOP/s.
7. Check correctness against a CPU reference for at least small and medium sizes.
8. Print parseable output.

Recommended command-line arguments:

1. matrix size `n` for square `n x n` matrices
2. measured iterations `iters`
3. warmup iterations `warmup`

## Performance Metric

For square GEMM `C = A * B` with `n x n` matrices, the standard operation count is:

- `2 * n^3` floating-point operations

If average kernel time is `t` seconds, then:

- `FLOP/s = 2 * n^3 / t`
- `GFLOP/s = (2 * n^3) / (t * 1e9)`

State clearly whether your timing includes:

1. GEMM only
2. host-to-device and device-to-host transfers
3. allocation and setup

Minimum requirement:

- report GEMM-only timing

Recommended:

- also report end-to-end timing to compare with GEMM-only timing

## Experimental Plan

Run your program on the same GPU for several matrix sizes.

Recommended sweep:

- `n = 256, 512, 1024, 2048, 4096`

If the GPU memory is limited, use smaller values and state that clearly.

For each size:

1. run at least 3 measured repeats
2. record average GEMM time
3. compute average GFLOP/s
4. record correctness error when checked

## ChatGPT Requirement

Use ChatGPT as a coding tool, but document what happened.

Minimum requirement:

1. Save the prompt or prompts you used.
2. Save the first generated code version.
3. Describe what was wrong, missing, or unclear in the generated code.
4. Describe what you changed before the code compiled and ran correctly.

Your report should address:

1. Did ChatGPT generate valid cuBLAS API usage immediately?
2. Did it handle memory layout correctly?
3. Did it produce correct timing methodology?
4. What manual debugging or correction was still required?

Do not submit AI-generated code as if it were unquestioned ground truth.

Example prompt:

```text
Write a CUDA C++ program that uses cuBLAS to multiply two square matrices on an
NVIDIA GPU with cublasSgemm. The matrix size n should come from the command
line. The program should allocate host and device memory, initialize the input
matrices, run warmup iterations, time the GEMM call for several measured
iterations, compute GFLOP/s using 2*n^3/time, copy the result back, and compare
against a CPU reference for correctness. Also provide a Makefile using nvcc and
-lcublas.
```

Example ChatGPT answer summary:

- includes `#include <cuda_runtime.h>` and `#include <cublas_v2.h>`
- uses `cudaMalloc`, `cudaMemcpy`, `cublasCreate`, and `cublasSgemm`
- parses command-line arguments such as `n`, `iters`, and `warmup`
- times the GEMM region, usually with CUDA events
- computes GFLOP/s from the measured time
- links with `-lcublas` in the build command

What you should still verify manually:

1. whether the matrix layout is row-major or column-major
2. whether the leading dimensions passed to `cublasSgemm` are correct
3. whether the timing includes only GEMM or also transfer/setup cost
4. whether the CPU correctness check matches the GPU layout convention

## Build Requirement

Create your own build in this assignment folder.

Minimum requirement:

1. a `Makefile`
2. a build command using `nvcc`
3. cuBLAS linking, such as `-lcublas`
4. a clear way to set the GPU architecture if needed

Example:

```bash
make
make CUDA_ARCH=sm_80
```

## Suggested Commands

These commands are examples and may need to be adapted for the target GPU.

```bash
make
./app 512 5 1
./app 1024 5 1
./app 2048 5 1
./app 4096 3 1
```

## Deliverables

Submit one folder containing:

1. Source code
   - GEMM program
   - Makefile
2. Results data
   - raw logs
   - one CSV file with at least these columns:
     - `n,iters,warmup,gemm_s,gflops,max_abs_err`
3. Report (`report_cublas_gemm.pdf`, 1-3 pages) including:
   - hardware used
   - CUDA version if known
   - compiler and flags
   - prompt(s) given to ChatGPT
   - major fixes needed after code generation
   - one plot of GFLOP/s vs matrix size
   - one short interpretation of the performance trend

## Grading Focus

1. Correct cuBLAS GEMM usage.
2. Correct FLOP/s calculation.
3. Reasonable timing methodology.
4. Clear matrix-size sweep and analysis.
5. Honest and specific discussion of the AI-generated starting point.

## Notes

1. Do not fabricate performance numbers.
2. Clearly state the GPU model if available.
3. If full GPU execution is unavailable, still submit code, build commands,
   ChatGPT prompts, and a short note describing what could not be run.
