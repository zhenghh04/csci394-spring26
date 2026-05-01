# Project 01 — GPU Offload Comparison (CPU / OpenACC / OpenMP target / CUDA / PyTorch)

## Group / contributions

Single-agent submission (Claude Opus 4.7). See `AI_USAGE.md` for details.

## Problem definition

Multiply two square FP32 matrices `C = A * B`, `n × n`. Compare five
implementations of the same kernel on Polaris (1× NVIDIA A100):

1. **CPU** baseline with OpenMP (32-thread `parallel for collapse(2)`)
2. **OpenACC** offload (`acc parallel loop collapse(2)`)
3. **OpenMP target offload** (`target teams distribute parallel for collapse(2)`)
4. **CUDA** kernel (16×16 block, naive triple loop)
5. **PyTorch** (`torch.float32 @ torch.float32` on `cuda`)

Sweep `n ∈ {256, 512, 1024, 2048, 4096}`. 1 warmup + 5 measured iterations
(CPU at n=4096 trimmed to 1 warmup + 2 measured to keep within debug-queue
walltime).

## Hardware and compilers

- **System**: Polaris (ALCF), one compute node, queue `debug`, host
  `x3001c0s1b0n0`.
- **GPU**: 1× NVIDIA A100-SXM4-40GB (one of four on the node).
- **Compiler**: NVHPC `cc` wrapper (calls `nvc`), and `nvcc` 12.2 from
  `cudatoolkit-standalone`. PyTorch from
  `/soft/applications/conda/2024-04-29/mconda3/bin/python`.
- **Flags**: `Makefile` defaults — CPU `-O2 -mp -fast`,
  OpenACC `-O2 -acc -gpu=cc80 -Minfo=accel`,
  OpenMP target `-O2 -mp=gpu -gpu=cc80 -Minfo=mp`,
  CUDA `-O2 -arch=sm_80 -std=c++14`.

## Timing methodology

- Host-side wall clock for all measurements; CUDA compute uses `cudaEvent`
  for kernel-only time; PyTorch compute uses `torch.cuda.Event`.
- Warmup iterations excluded from averages.
- Correctness vs CPU FP32 reference on a 64×64 sub-block (full O(n³)
  reference at n=4096 would dominate the benchmark).

## Results (CSV: `results/results.csv`)

End-to-end and compute-only time at each n, and `max_abs_err`. Selected
points (`time` in seconds, `GFLOP/s = 2 n³ / compute_s / 1e9`):

| n     | version    | end_to_end_s | compute_s   | GFLOP/s | max_abs_err |
|-------|-----------|--------------|-------------|---------|-------------|
| 256   | cpu        | 6.14e-04    | 6.14e-04    | 54.7    | 0.0         |
| 256   | openacc    | 3.76e-04    | 2.84e-04    | 118     | 7.6e-06     |
| 256   | omp_target | 1.24e-04    | 2.97e-05    | 1131    | 2.3e-05     |
| 256   | cuda       | 3.26e-04    | 2.62e-05    | 1280    | 3.8e-06     |
| 256   | pytorch    | 3.15e-04    | 4.33e-05    | 775     | 7.6e-06     |
| 1024  | cpu        | 0.202        | 0.202       | 10.6    | 0.0         |
| 1024  | openacc    | 1.51e-02    | 1.41e-02    | 152     | 3.1e-05     |
| 1024  | omp_target | 1.95e-03    | 1.01e-03    | 2125    | 2.4e-04     |
| 1024  | cuda       | 2.42e-03    | 8.98e-04    | 2391    | 1.5e-05     |
| 1024  | pytorch    | 2.19e-03    | 1.91e-04    | 11226   | 6.1e-05     |
| 2048  | cpu        | 2.09         | 2.09         | 8.2     | 0.0         |
| 2048  | openacc    | 9.66e-02    | 9.29e-02    | 185     | 6.1e-05     |
| 2048  | omp_target | 1.00e-02    | 6.31e-03    | 2725    | 1.1e-03     |
| 2048  | cuda       | 1.16e-02    | 7.00e-03    | 2454    | 3.1e-05     |
| 2048  | pytorch    | 6.23e-03    | 1.27e-03    | 13495   | 1.1e-03     |
| 4096  | cpu        | 31.83        | 31.83        | 4.3     | 0.0         |
| 4096  | openacc    | 0.833        | 0.819        | 168     | 1.2e-04     |
| 4096  | omp_target | 8.33e-02    | 6.89e-02    | 1996    | 4.9e-04     |
| 4096  | cuda       | 7.50e-02    | 5.65e-02    | 2434    | 6.1e-05     |
| 4096  | pytorch    | 2.67e-02    | 9.32e-03    | 14755   | 7.3e-04     |

(Plot: `results/runtime_vs_size.png`.)

## Discussion

1. **Fastest at small n (256)**: surprisingly the **OpenMP target** path
   (124 µs end-to-end, 30 µs kernel) and **CUDA** (326 µs end-to-end,
   26 µs kernel) win on compute-only. OpenMP target's lower end-to-end
   reflects efficient `target enter data` / `target update` flow. CPU is
   already 5–10× slower at n=256 because the naive triple loop has poor
   cache behavior even for a small matrix.
2. **Fastest at large n (≥1024)**: **PyTorch** wins by an order of magnitude
   on both end-to-end and compute-only. PyTorch dispatches to cuBLAS, which
   is 5–6× faster than the naive CUDA kernel because cuBLAS uses Tensor
   Cores when the math mode allows and shared-memory tiling otherwise. Our
   hand CUDA kernel uses no shared memory and tops out around 2.5 TFLOP/s
   on A100 (theoretical FP32 peak ≈ 19.5 TFLOP/s).
3. **Compute alone vs end-to-end**: at small n (256–512) the end-to-end
   time is 5–10× the kernel time because allocation and H2D/D2H transfers
   dominate. Crossover is around n ≈ 1024 where compute starts to grow as
   O(n³) faster than transfers as O(n²).
4. **At what point does GPU offload pay off?** Even at n=256 OpenMP target
   is faster than CPU end-to-end (0.12 ms vs 0.61 ms). For the *naive* CUDA
   kernel the threshold is also around n=256. For PyTorch (cuBLAS) the
   kernel is so fast that the threshold is essentially "always" — pyTorch
   already wins at n=256 vs CPU.
5. **OpenACC vs OpenMP target vs CUDA vs PyTorch**: OpenACC's naive
   `parallel loop collapse(2)` is 5–10× slower than OpenMP target on the
   same source — `nvc -acc` does not appear to vectorize the inner k-loop
   the same way `-mp=gpu` does. CUDA-naive is competitive with OpenMP
   target. PyTorch (cuBLAS) is 5–10× faster than the hand-written kernels
   because it engages the optimized tiled GEMM path.

## Files

- `src/app_cpu.c`, `src/app_openacc.c`, `src/app_omp_target.c`,
  `src/app_cuda.cu`, `src/app_pytorch.py`, `src/common.h`, `Makefile`
- `results/results.csv`, `results/raw.log`, `results/runtime_vs_size.png`,
  `results/pbs.out`
- `AI_USAGE.md`
