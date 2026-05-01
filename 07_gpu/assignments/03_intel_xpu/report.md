# Project 03 — Intel XPU GEMM on Aurora (FP32 + XMX joint_matrix BF16→FP32)

## Group / contributions

Single-agent submission (Claude Opus 4.7). See `AI_USAGE.md`.

## Hardware

- **System**: Aurora (ALCF), one compute node, queue `debug`, host
  `x4217c5s2b0n0`. Project allocation: `CSCI394-HPC`.
- **GPU**: 1× **Intel Data Center GPU Max 1550** (Ponte Vecchio / "PVC"),
  one of six on the node.
- **Compiler**: Intel oneAPI DPC++/C++ Compiler 2025.3.2 (icpx),
  oneAPI 2025.3.1 module on Aurora 26.26.0.
- **Build**: `icpx -fsycl -O2 -fsycl-targets=spir64 -o app_xpu src/app_xpu.cpp`.
- **Runtime selector**: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (force the
  level_zero backend so `joint_matrix` is engaged).

## Modes

- `fp32`           — naive USM-device SYCL `parallel_for` GEMM kernel
                      (no shared local memory tiling). Baseline only.
- `xmx_bf16_fp32`  — `joint_matrix` tiled GEMM, tile shape (TM,TN,TK) =
                     (8,16,16), sub-group size 16. BF16 inputs on the
                     `use::a` and `use::b` matrices, FP32 accumulator,
                     FP32 store.

## Results (CSV: `results/results.csv`)

| n    | mode             | time_s    | GFLOP/s | max_abs_err |
|------|------------------|-----------|---------|-------------|
| 256  | fp32             | 1.44e-04 |   233   | 3.1e-05     |
| 256  | xmx_bf16_fp32    | 2.56e-05 |  1313   | 1.0e-01     |
| 512  | fp32             | 5.65e-04 |   475   | 4.6e-05     |
| 512  | xmx_bf16_fp32    | 9.00e-05 |  2983   | 2.0e-01     |
| 1024 | fp32             | 5.02e-03 |   428   | 2.3e-04     |
| 1024 | xmx_bf16_fp32    | 3.76e-04 |  5718   | 4.1e-01     |
| 2048 | fp32             | 5.75e-02 |   299   | 1.1e-03     |
| 2048 | xmx_bf16_fp32    | 5.09e-03 |  3374   | 8.0e-01     |
| 4096 | fp32             | 4.79e-01 |   287   | 4.9e-04     |
| 4096 | xmx_bf16_fp32    | 5.04e-02 |  2726   | 1.6        |

(Plot: `results/gflops_vs_size.png`.)

## Discussion

1. **FP32 throughput vs n**: the naive `parallel_for` kernel rises from
   233 GFLOP/s at n=256 to 475 GFLOP/s at n=512 then *drops* back to
   ~290 GFLOP/s at n≥2048. This is the classic global-memory-bound
   curve — without SLM tiling the kernel reuses A/B from main memory for
   every k-step and thrashes the L1/L2 caches. PVC's FP32 peak is on the
   order of 50 TFLOP/s; we are reaching well under 1% of peak. **The
   point of this baseline is not to be fast — it is to motivate XMX.**
2. **XMX throughput vs n**: peaks at 5.7 TFLOP/s (n=1024) then declines.
   This is *also* well below PVC's peak XMX BF16 throughput (≈ 100–300
   TFLOP/s depending on sparsity / cooperative fetch). The reason is that
   our `joint_matrix` kernel uses a single sub-group per output tile and
   loads A and B from global memory directly each k-step; there is no
   local-memory blocking, no multi-sub-group cooperation, and no
   pipelined prefetch. To reach PVC peak you need a tiled cooperative
   load pattern (or call oneMKL `gemm`). The peak at n=1024 reflects the
   sweet spot where the working set fits in L2 but the launch overhead is
   amortized.
3. **Gap**: at n=256 XMX is 5.6× faster than naive FP32; at n=1024 it is
   13×; at n=4096 it is ~9.5×. The XMX path always wins, but neither
   path is close to peak — a fair "tuned" comparison would use oneMKL.
4. **Precision mode**: BF16 inputs, FP32 accumulator, FP32 store. Mixed
   precision, *not* pure FP32 arithmetic.
5. **Accuracy tradeoff**: `max_abs_err` for `xmx_bf16_fp32` grows from
   ≈0.1 at n=256 to ≈1.6 at n=4096 because each multiply rounds to BF16
   (~3 decimal digits) and the larger n the more rounding accumulates
   even though the partial sums stay in FP32. The pure FP32 path stays
   below 1e-3 across all n. For ML this is acceptable; for numerics it
   is not.
6. **AI quality**: the AI-generated `joint_matrix` code initially used the
   removed `experimental` API path with `get_wi_data`; it had to be
   updated to the unified `joint_matrix_load`/`mad`/`store` interface.
   Pointer space casting (`address_space_cast<global_space, no>`) was
   missing in the first draft. The `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
   override was *not* in the AI's first draft — without it the kernel
   silently falls back to OpenCL (no `joint_matrix` support) and segfaults
   on launch (which is exactly what happened on the v1 attempt — see
   `core.NNNNN` files in the original Aurora workdir). See `AI_USAGE.md`.

## Files

- `src/app_xpu.cpp`, `Makefile`
- `results/results.csv`, `results/raw.log`,
  `results/gflops_vs_size.png`, `results/pbs.out`
- `AI_USAGE.md`
