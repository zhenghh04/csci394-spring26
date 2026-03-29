# GPU Assignment: Dense Matrix Multiplication Across CPU, OpenACC, OpenMP Target Offload, and CUDA

## Objective

Implement and compare the same dense matrix-matrix multiplication kernel in
four versions:

1. CPU baseline with OpenMP
2. OpenACC offload
3. OpenMP target offload
4. CUDA

Then measure correctness and performance to answer a central Chapter 16-18
question:

- when does GPU offload help enough to justify host-device transfer overhead?

## Kernel Definition

Use square dense matrix multiplication:

- `C = A * B`
- `A`, `B`, and `C` are all `n x n`
- use FP32 unless you clearly document another choice

For a simple reference implementation, the CPU version may use the standard
triple loop:

```c
for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            C[i*n + j] += A[i*n + k] * B[k*n + j];
```

## Required Code Changes

Create your own program(s) in this assignment folder.

Minimum requirement:

1. one CPU implementation
2. one OpenACC implementation
3. one OpenMP target offload implementation
4. one CUDA implementation
5. one correctness check against the CPU result
6. parseable timing output
7. command-line input for matrix size `n`

Recommended command-line arguments:

1. matrix size `n`
2. measured iterations `iters`
3. warmup iterations `warmup`

## Timing Requirements

For each version, measure the timing for following part
   - end-to-end time: data transfer + compute
   - data transfer if applicable
   - compute
   - copy-back if applicable

Use warmup runs and exclude warmup from measured statistics.

## Performance Metric

For square matrix multiplication `C = A * B` with `n x n` matrices, use:

- operations = `2 * n^3`

If average compute time is `t` seconds, you may also report:

- `FLOP/s = 2 * n^3 / t`
- `GFLOP/s = (2 * n^3) / (t * 1e9)`

Minimum requirement:

- report runtime in seconds

Recommended:

- also report GFLOP/s for the compute-region timing

## Experimental Plan

Run all four versions on the same machine if possible and compare them across
problem sizes.

Recommended sweep:

- `n = 256, 512, 1024, 2048, 4096`

If your GPU memory is limited, use smaller values and state that clearly.

For each problem size:

1. run at least 5 repeats
2. record average end-to-end time
3. record average compute-region time
4. record maximum absolute error

## What to Analyze

Your report should answer:

1. Which version is fastest for small problem sizes?
2. Which version is fastest for large problem sizes?
3. Does GPU compute time alone tell the whole story?
4. At what point does transfer overhead stop dominating?
5. How do OpenACC, OpenMP target offload, and CUDA compare on the same GEMM?

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
   - examples: `ChatGPT`, `Claude`, `Gemini`, `AskALCF`
2. the model name if known
   - examples: `GPT-5.4`, `Claude 3.7 Sonnet`
3. the exact prompt or prompts you used
4. what code, explanation, or debugging help the AI provided
5. what you changed after receiving the AI output

If you use online help that is not AI, you must state:

1. the site or source
2. what information you used from it
3. how it influenced your implementation

Do not hide AI usage or present generated code as if it appeared without
assistance.

## Build Requirement

Create your own build in this assignment folder.

Minimum requirement:

1. build target for CPU version
2. build target for OpenACC version
3. build target for OpenMP target offload version
4. build target for CUDA version
5. clear comments or variables showing how to enable GPU compilation flags on
   the target system

## Suggested Commands

These commands are examples and may need to be adapted for the actual cluster or
compiler.

CPU baseline:

```bash
make cpu
./app_cpu 512 5 1
```

OpenACC:

```bash
make openacc ACCFLAGS='-acc -Minfo=accel'
./app_openacc 512 5 1
```

OpenMP target offload:

```bash
make omp_target OFFLOAD_FLAGS='<site-specific target flags>'
./app_omp_target 512 5 1
```

CUDA:

```bash
make cuda
./app_cuda 512 5 1
```

## Deliverables

Submit one folder containing:

1. Source code
   - CPU version
   - OpenACC version
   - OpenMP target offload version
   - CUDA version
   - Makefile
2. Results data
   - raw logs
   - one CSV file with at least these columns:
     - `version,n,iters,warmup,end_to_end_s,compute_s,max_abs_err`
3. Report (`report_gpu_offload.pdf`, 1-3 pages) including:
   - problem definition
   - timing method
   - hardware and compiler used
   - AI tools or online help used, if any
   - AI model name, if any
   - prompt(s) used with the AI tool, if any
   - one runtime-vs-size plot
   - one short discussion of transfer overhead
   - one short comparison of OpenACC vs OpenMP target offload vs CUDA

## Grading Focus

1. correctness of all four implementations
2. fair and reproducible timing methodology
3. clear comparison across problem sizes
4. accurate reasoning about host-device transfer costs
5. clear presentation of plots and data

## Notes

1. Do not fabricate timing numbers.
2. Clearly state compiler flags and hardware used.
3. If real GPU execution is unavailable, state that explicitly and still provide
   the CPU correctness path plus build commands for the GPU versions.
4. If you used AI or online help, document that use specifically and honestly.

## References

1. OpenMP specifications
   - https://www.openmp.org/specifications/
2. OpenMP target teams distribute parallel for construct
   - https://www.openmp.org/spec-html/5.1/openmpsu97.html
3. OpenMP target teams distribute parallel for simd construct
   - https://www.openmp.org/spec-html/5.1/openmpsu98.html
4. OpenACC specification
   - https://www.openacc.org/specification
5. CUDA Programming Guide
   - https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
6. cuBLAS documentation
   - https://docs.nvidia.com/cuda/cublas/
