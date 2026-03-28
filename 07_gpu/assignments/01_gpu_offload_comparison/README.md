# GPU Assignment: CPU vs OpenACC vs OpenMP Target Offload

## Objective

Implement and compare the same numerical kernel in three versions:

1. CPU baseline
2. OpenACC offload
3. OpenMP target offload
4. CUDA

Then measure correctness and performance to answer a central Chapter 16-18
question:

- when does GPU offload help enough to justify host-device transfer overhead?

## Recommended Starting Points

Use the existing course examples as references:

- OpenACC:
  - `../../openacc/01_parallel_loop/main.c`
  - `../../openacc/02_kernels_matvec/main.c`
  - `../../openacc/03_data_movement/main.c`
- OpenMP target offload:
  - `../../openmp/16_target_offload/target_axpy.c`
  - `../../openmp/16_target_offload/target_matmul.c`

You may choose **one** of these kernels for the assignment:

1. AXPY / vector update
   - `out[i] = a * x[i] + y[i]`
2. Dense matrix-vector multiply
   - `y = A * x`

Recommended choice:

- dense matrix-vector multiply, because data movement and arithmetic work are both
  easy to reason about

## Required Code Changes

Create your own program(s) in this assignment folder.

Minimum requirement:

1. One CPU implementation.
2. One OpenACC implementation.
3. One OpenMP target offload implementation.
4. One correctness check against the CPU result.
5. Parseable timing output.

If you choose AXPY:

- support command-line input for vector size `n`

If you choose matrix-vector multiply:

- support command-line input for matrix size `n`
- matrix is `n x n`

## Timing Requirements

For each version, measure:

1. End-to-end time
   - allocation
   - initialization
   - data transfer if applicable
   - compute
   - copy-back if applicable
2. Compute-region time
   - kernel-only time if possible
3. Correctness
   - maximum absolute error against CPU reference

Use warmup runs and exclude warmup from measured statistics.

Suggested command-line arguments:

1. problem size (`n`)
2. measured iterations (`iters`)
3. warmup iterations (`warmup`)

## Experimental Plan

Run the three versions on the same machine and compare them across problem sizes.

Recommended sweep:

- `n = 2^10, 2^12, 2^14, 2^16, 2^18, 2^20` for AXPY

or

- `n = 256, 512, 1024, 2048, 4096` for matrix-vector multiply

For each problem size:

1. run at least 3 repeats
2. record average end-to-end time
3. record average compute-region time
4. record maximum absolute error

## What to Analyze

Your report should answer:

1. Which version is fastest for small problem sizes?
2. Which version is fastest for large problem sizes?
3. Does GPU compute time alone tell the whole story?
4. At what point does transfer overhead stop dominating?
5. How do OpenACC and OpenMP target offload compare on the same problem?

## Build Requirement

Create your own build in this assignment folder.

Minimum requirement:

1. Build target for CPU version.
2. Build target for OpenACC version.
3. Build target for OpenMP target offload version.
4. Clear comments or variables showing how to enable GPU compilation flags on the
   target system.

## Suggested Commands

These commands are examples and may need to be adapted for the actual cluster or
compiler.

CPU baseline:

```bash
make cpu
./app_cpu 1048576 5 1
```

OpenACC:

```bash
make openacc ACCFLAGS='-acc -Minfo=accel'
./app_openacc 1048576 5 1
```

OpenMP target offload:

```bash
make omp_target OFFLOAD_FLAGS='<site-specific target flags>'
./app_omp_target 1048576 5 1
```

## Deliverables

Submit one folder containing:

1. Source code
   - CPU version
   - OpenACC version
   - OpenMP target offload version
   - Makefile
2. Results data
   - raw logs
   - one CSV file with at least these columns:
     - `version,n,iters,warmup,end_to_end_s,compute_s,max_abs_err`
3. Report (`report_gpu_offload.pdf`, 1-3 pages) including:
   - problem definition
   - timing method
   - hardware and compiler used
   - one runtime-vs-size plot
   - one short discussion of transfer overhead
   - one short comparison of OpenACC vs OpenMP target offload

## Grading Focus

1. Correctness of all three implementations.
2. Fair and reproducible timing methodology.
3. Clear comparison across problem sizes.
4. Accurate reasoning about host-device transfer costs.
5. Clear presentation of plots and data.

## Notes

1. Do not fabricate timing numbers.
2. Clearly state compiler flags and hardware used.
3. If real GPU execution is unavailable, state that explicitly and still provide the
   CPU correctness path plus build commands for the GPU versions.
