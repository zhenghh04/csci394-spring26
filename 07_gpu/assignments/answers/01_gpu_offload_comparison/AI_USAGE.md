# AI usage statement — Project 01 (GPU offload comparison)

- **Product**: Claude Code (CLI in VSCode)
- **Model**: Claude Opus 4.7 (1M context), `claude-opus-4-7`
- **Date**: 2026-04-28

## Prompt (verbatim)

> Let us do the assignments for csci394-spring26 07_gpu/assignments/ Please do all the three

(One short instruction; no per-project prompts. The agent expanded this into the
implementations, build files, run scripts, and reports.)

## What the AI produced

For Project 01 it generated:

1. `src/common.h` — host-side matrix init and a sampled CPU reference.
2. `src/app_cpu.c` — OpenMP CPU baseline with timed warmup/measured loop.
3. `src/app_openacc.c` — OpenACC offload with separate H2D / kernel / D2H timers.
4. `src/app_omp_target.c` — OpenMP `target teams distribute parallel for` GEMM
   with `enter data` / `target update` style transfer measurement.
5. `src/app_cuda.cu` — CUDA kernel with cudaEvent-based kernel timing and
   `chrono`-based H2D / D2H timing.
6. `src/app_pytorch.py` — Torch `@` operator on GPU with `cuda.Event` for
   compute timing and `cpu()` round trip for transfer time.
7. `Makefile` with NVHPC-compatible defaults (`nvc`, `nvcc`, `cc80`).
8. A Polaris job script workflow that emits a child PBS script for the `debug`
   queue, waits for the PBS job, and plots `results.csv`.

## What was changed / verified manually

- Forced reproducible matrix init pattern (same across all 5 versions) so the
  correctness check is meaningful across implementations.
- Sampled the correctness check to a 64×64 block — naive O(n³) reference at
  n=4096 on the CPU would take longer than the actual benchmark.
- Used CUDA events (not host wall clock) for the CUDA compute-only timing.
- Capped the CPU baseline at n=2048 with full iters; n=4096 runs only 2 iters.
- Ran the benchmark through a generated PBS job on Polaris after the direct IRI
  API path failed with an expired/truncated token.

## What did not work first try / lessons

- The PyTorch `cuda.Event.elapsed_time` returns ms, not s — explicitly divided
  by 1e3.
- `nvc` requires `-mp=gpu -gpu=cc80` for OpenMP target offload (not bare
  `-fopenmp`); same compiler also handles `-acc`.
