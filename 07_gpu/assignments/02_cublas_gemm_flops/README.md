# GPU Assignment: Measure FP32 and Tensor Core GEMM FLOPs with cuBLAS

## Objective

Use NVIDIA cuBLAS to measure dense matrix-multiplication performance on a GPU
for two different hardware paths:

1. standard FP32 GEMM throughput
2. tensor core GEMM throughput

The central questions are:

- how does measured FP32 GEMM throughput change with matrix size?
- how much higher is the effective throughput when tensor cores are used?
- what precision tradeoffs appear in the tensor core path?

This assignment also asks you to use AI and online help carefully and document
what happened.

## Technical Scope

Your code should target an NVIDIA GPU using CUDA and cuBLAS.

You must implement or generate two GPU paths:

1. **FP32 GEMM path**
   - use FP32 inputs and FP32 output
   - recommended API: `cublasSgemm`
2. **Tensor core GEMM path**
   - use a tensor-core-capable cuBLAS path
   - acceptable choices include:
     - `cublasGemmEx`
     - `cublasLtMatmul`
   - common mixed-precision choices include:
     - FP16 input with FP32 accumulation
     - BF16 input with FP32 accumulation
     - TF32 on supported GPUs

Recommended interpretation:

- treat the FP32 path as the conventional floating-point baseline
- treat the tensor core path as the accelerated matrix-engine measurement
- report the tensor core result as **effective FLOP/s** using the standard GEMM
  operation count `2*n^3 / time`, while clearly stating the operand precision

## Brief Background: Tensor Cores

Tensor cores are specialized matrix-math hardware units on modern NVIDIA GPUs.

At a high level:

- regular FP32 GEMM uses the standard floating-point datapath
- tensor core GEMM uses specialized hardware for matrix multiply-accumulate
- tensor core paths are often mixed precision
- this can produce much higher throughput, but sometimes with different
  numerical accuracy

For this assignment, the main comparison is:

1. a normal FP32 cuBLAS GEMM path
2. a tensor-core-enabled GEMM path

## Required Program Features

Create your own program in this assignment folder.

Minimum requirement:

1. accept matrix size `n` from the command line
2. accept measured iterations `iters`
3. accept warmup iterations `warmup`
4. allocate matrices on the host and device
5. initialize input matrices with a reproducible pattern
6. run one FP32 GEMM path on the GPU
7. run one tensor core GEMM path on the GPU
8. time the GEMM region for both paths
9. report the achieved FLOP/s or GFLOP/s
10. check correctness against a trusted reference
11. print parseable output

Recommended output columns:

- `mode,n,iters,warmup,gemm_s,gflops,max_abs_err`

Recommended `mode` values:

- `fp32`
- `tensor_core`

## Correctness Requirement

You must compare against a trusted reference.

Acceptable choices:

1. CPU FP32 GEMM for small and medium sizes
2. FP32 cuBLAS GEMM used as the reference when validating the tensor core path

Minimum requirement:

- verify correctness for at least small and medium sizes
- report `max_abs_err`
- clearly state whether the tensor core path is mixed precision

Because tensor core paths often use FP16, BF16, or TF32 input behavior, you may
observe larger numerical error than the pure FP32 path. That is acceptable if
you explain it clearly.

## Performance Metric

For square GEMM `C = A * B` with `n x n` matrices, the standard operation count
is:

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

## Allowed Help

You are allowed to use:

1. AI tools
2. online documentation
3. online examples
4. forum posts or other web resources

This assignment does **not** require you to work without help. It does require
you to document your help sources honestly.

If you use AI, you must state:

1. the product name
   - examples: `ChatGPT`, `Claude`, `Gemini`
2. the model name if known
   - example: `GPT-5.4`
3. the exact prompt or prompts you used
4. what code, explanation, or debugging help the AI provided
5. what you changed after receiving the AI output

If you use online help that is not AI, you must state:

1. the site or source
2. what information you used from it
3. how it influenced your implementation

Do not hide AI usage or present generated code as if it appeared without
assistance.

## AI Guidance

Use AI as a coding tool, but document what happened.

Your report should address:

1. Did the AI generate valid cuBLAS API usage immediately?
2. Did it use a correct tensor core path?
3. Did it handle memory layout correctly?
4. Did it produce correct timing methodology?
5. What manual debugging or correction was still required?

Do not submit AI-generated code as if it were unquestioned ground truth.

Example prompt for the FP32 path:

```text
Write a CUDA C++ program that uses cuBLAS to multiply two square matrices on an
NVIDIA GPU with cublasSgemm. The matrix size n should come from the command
line. The program should allocate host and device memory, initialize the input
matrices, run warmup iterations, time the GEMM call for several measured
iterations, compute GFLOP/s using 2*n^3/time, copy the result back, and compare
against a CPU reference for correctness. Also provide a Makefile using nvcc and
-lcublas.
```

Example prompt for the tensor core path:

```text
Write a CUDA C++ program that uses cuBLAS on an NVIDIA GPU to measure tensor
core GEMM performance for square matrices. Use either cublasGemmEx or
cublasLtMatmul, accept n, iters, and warmup from the command line, time the
GEMM region only, compute effective GFLOP/s using 2*n^3/time, and compare the
result against an FP32 reference. Clearly state the precision mode used and
provide a Makefile using nvcc and -lcublas.
```

What you should still verify manually:

1. whether the matrix layout is row-major or column-major
2. whether the leading dimensions passed to cuBLAS are correct
3. whether the tensor core path really uses the intended math mode or datatype
4. whether the timing includes only GEMM or also transfer/setup cost
5. whether the CPU correctness check matches the GPU layout convention

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

If you build separate executables:

```bash
make
./app_fp32 512 5 1
./app_tensor 512 5 1
./app_fp32 2048 5 1
./app_tensor 2048 5 1
```

If you build one combined executable, that is also acceptable:

```bash
./app fp32 1024 5 1
./app tensor 1024 5 1
```

## What to Analyze

Your report should answer:

1. How does FP32 GEMM throughput change with matrix size?
2. How does tensor core GEMM throughput change with matrix size?
3. How large is the gap between the FP32 path and the tensor core path?
4. What precision mode was used for the tensor core path?
5. What numerical accuracy tradeoff did you observe?
6. What mistakes did the AI tool make in the initial code?

## Deliverables

Submit one folder containing:

1. Source code
   - FP32 GEMM program or mode
   - tensor core GEMM program or mode
   - Makefile
2. Results data
   - raw logs
   - one CSV file with at least these columns:
     - `mode,n,iters,warmup,gemm_s,gflops,max_abs_err`
3. Report (`report_cublas_gemm.pdf`, 1-3 pages) including:
   - hardware used
   - CUDA version if known
   - compiler and flags
   - AI tools or online help used, if any
   - AI model name, if any
   - prompt(s) used with the AI tool, if any
   - major fixes needed after code generation
   - one plot of GFLOP/s vs matrix size
   - one short interpretation of FP32 vs tensor core performance

## Grading Focus

1. Correct cuBLAS GEMM usage.
2. Correct FLOP/s calculation.
3. Clear distinction between FP32 baseline and tensor core measurement.
4. Reasonable timing methodology.
5. Honest and specific discussion of the AI-generated starting point.

## Notes

1. Do not fabricate performance numbers.
2. Clearly state the GPU model if available.
3. Clearly state the precision mode of the tensor core path.
4. If full GPU execution is unavailable, still submit code, build commands,
   prompts, and a short note describing what could not be run.

## References

Use the official NVIDIA documentation as the primary API reference:

1. cuBLAS documentation
   - https://docs.nvidia.com/cuda/cublas/
2. cuBLAS contents page
   - https://docs.nvidia.com/cuda/archive/12.6.0/cublas/contents.html
3. `cublasSgemm` API reference
   - https://docs.nvidia.com/cuda/cublas/#cublassgemm
4. `cublasGemmEx` API reference
   - https://docs.nvidia.com/cuda/cublas/#cublasgemmex
5. `cublasLtMatmul` API reference
   - https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
6. cuBLAS compute type reference
   - https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t
