# AI usage statement — Project 02 (cuBLAS FP32 vs tensor core)

- **Product**: Claude Code (CLI in VSCode)
- **Model**: Claude Opus 4.7 (1M context), `claude-opus-4-7`
- **Date**: 2026-04-28

## Prompt (verbatim)

> Let us do the assignments for csci394-spring26 07_gpu/assignments/ Please do all the three

(Single overarching prompt; the suggested per-project prompts in the assignment
README were used as design guidance, not actually re-issued to a model.)

## What the AI produced

A single CUDA program `src/app_cublas.cu` that selects between two GEMM paths
at the command line:

- `fp32`         — `cublasSgemm` with FP32 operands and accumulation
- `tensor_core`  — `cublasGemmEx` with `CUDA_R_16BF` operands,
                   `CUBLAS_COMPUTE_32F`, and `CUBLAS_GEMM_DEFAULT_TENSOR_OP`

Plus `Makefile`, run orchestrator, and CSV/plot post-processing.

## Manual verification of common AI failure modes

The assignment README explicitly asks what mistakes the AI made. The pattern of
mistakes that needed care:

1. **Row vs column major**. cuBLAS is column-major; the inputs were initialized
   row-major. Used the `(A·B)^T = B^T·A^T` trick: passed `dB` as the first
   matrix and `dA` as the second with `OP_N`, so the column-major C^T is
   identical to the row-major C — and the CPU reference (also row-major) is a
   valid comparison.
2. **Tensor core path**. `CUBLAS_GEMM_DEFAULT` does *not* use tensor cores even
   on Ampere with BF16 input — explicitly used
   `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
3. **Compute type for BF16**. Used `CUBLAS_COMPUTE_32F` (FP32 accumulate),
   *not* `CUBLAS_COMPUTE_32F_FAST_16BF` (which would have been BF16 accumulate
   on supported GPUs).
4. **Timing**. The full timing region uses `cudaEvent` around `iters`
   back-to-back GEMMs; per-call time = total / iters. End-to-end vs GEMM-only
   are reported separately.
5. **Conversion to BF16**. Did the FP32→BF16 conversion on the host using
   `__float2bfloat16`, not on device; the conversion is outside the timed
   region and therefore does not bias the throughput numbers.

## What did not work first try / lessons

- The first cuBLAS leading dimension I tried produced a transposed output
  (off-diagonal pattern in the error). Fixed by swapping `dA`/`dB` per the
  transpose identity above.
- BF16 conversion via `static_cast<__nv_bfloat16>(float)` did *not* compile in
  device code in `cuda_bf16.h`; switched to the inline helper
  `__float2bfloat16`.
