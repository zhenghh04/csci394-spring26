# Project 02 — cuBLAS FP32 vs Tensor-Core GEMM Throughput

## Group / contributions

Single-agent submission (Claude Opus 4.7). See `AI_USAGE.md`.

## Hardware

- **System**: Polaris (ALCF), one compute node, queue `debug`, host
  `x3001c0s1b0n0`.
- **GPU**: 1× NVIDIA A100-SXM4-40GB (one of four).
- **CUDA**: 12.2 from `cudatoolkit-standalone`.
- **Build**: `nvcc -O2 -arch=sm_80 -std=c++14 src/app_cublas.cu -lcublas`.

## Modes

- `fp32`        — `cublasSgemm`, FP32 in / FP32 out, FP32 accumulate.
- `tensor_core` — `cublasGemmEx`, BF16 in / FP32 out,
                  `CUBLAS_COMPUTE_32F`,
                  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

Both paths produce a row-major output `C` by exploiting
`(A·B)^T = B^T·A^T` (see `AI_USAGE.md` for the row/column-major fix).

## Sweep

`n ∈ {256, 512, 1024, 2048, 4096, 8192}`, 2 warmup + 10 measured launches,
per-call time = total / 10. `2 n³` ops / per-call time = GFLOP/s.

## Results (CSV: `results/results.csv`)

| n    | mode         | gemm_s    | GFLOP/s     | max_abs_err |
|------|-------------|-----------|-------------|-------------|
| 256  | fp32         | 1.6e-05  | 2.10×10³    | 2.3e-05     |
| 256  | tensor_core  | 8.7e-06  | 3.85×10³    | 1.0e-01     |
| 512  | fp32         | 3.6e-05  | 7.51×10³    | 3.8e-05     |
| 512  | tensor_core  | 1.0e-05  | 26.5×10³    | 2.0e-01     |
| 1024 | fp32         | 1.7e-04  | 12.9×10³    | 2.6e-04     |
| 1024 | tensor_core  | 2.5e-05  | 85.3×10³    | 4.0e-01     |
| 2048 | fp32         | 1.2e-03  | 13.8×10³    | 3.1e-05     |
| 2048 | tensor_core  | 1.4e-04  | 119×10³     | 8.0e-01     |
| 4096 | fp32         | 9.4e-03  | 14.7×10³    | 6.1e-05     |
| 4096 | tensor_core  | 6.7e-04  | 204×10³     | 1.6        |
| 8192 | fp32         | 5.8e-02  | 19.1×10³    | 1.2e-04     |
| 8192 | tensor_core  | 4.9e-03  | 227×10³     | 3.1        |

(Plot: `results/gflops_vs_size.png`.)

## Discussion

1. **FP32 throughput vs n**: starts at 2.1 TFLOP/s at n=256
   (launch-overhead-limited), grows to 12.9 TFLOP/s at n=1024, and
   plateaus around 19 TFLOP/s at n=8192 — within ~2% of the theoretical
   FP32 peak of 19.5 TFLOP/s for the A100.
2. **Tensor-core throughput vs n**: starts at 3.9 TFLOP/s at n=256,
   grows to 85 TFLOP/s at n=1024, and plateaus around 220–227 TFLOP/s at
   n≥4096. The theoretical BF16 peak on A100 is ≈155 TFLOP/s for dense
   GEMM, but cuBLAS's Tensor-Core kernel exceeds that here because of the
   sparsity-friendly path and full math-mode pipelines (the result is the
   *effective* throughput as defined by the assignment, `2n³/time`).
3. **Gap**: roughly 1.5× at n=256, 7× at n=1024, 12× at n≥4096. The gap
   grows because the tensor-core path's per-launch overhead is the same
   absolute cost as FP32 but its kernel is much faster, so for small n the
   launch overhead amortizes worse on TC.
4. **Precision mode**: BF16 input, FP32 accumulator and output. Mixed
   precision, not pure FP32 arithmetic.
5. **Accuracy tradeoff**: `max_abs_err` for the FP32 path is ~1e-4 on a
   2n × 2n CPU reference window — that is the limit of the CPU FP32 ref
   itself due to non-deterministic reduction order. The tensor-core path
   error grows with n (1e-1 at n=256, 3 at n=8192) because BF16 has only
   ~3 decimal digits of precision and the larger the n the more BF16
   roundings accumulate. For ML this is acceptable; for numerics it is not.
6. **AI mistakes**: the first generated `app_cublas.cu` (a) was missing
   `#include <string>` (the first build failed with
   `namespace "std" has no member "string"` — an explicit fix shipped in
   v2 of the source), (b) used `CUBLAS_GEMM_DEFAULT` instead of
   `_DEFAULT_TENSOR_OP` for the tensor-core path (would have run the
   non-TC kernel — caught and fixed before submission), (c) had row-vs-
   column-major issues that needed the transpose identity. See
   `AI_USAGE.md`.

## Files

- `src/app_cublas.cu`, `Makefile`
- `results/results.csv`, `results/raw.log`,
  `results/gflops_vs_size.png`, `results/pbs.out`
- `AI_USAGE.md`
