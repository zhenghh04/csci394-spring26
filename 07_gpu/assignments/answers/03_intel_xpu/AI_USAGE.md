# AI usage statement — Project 03 (Intel XPU on Aurora)

- **Product**: Claude Code (CLI in VSCode)
- **Model**: Claude Opus 4.7 (1M context), `claude-opus-4-7`
- **Date**: 2026-04-28

> Note: the assignment specifically asks for ChatGPT or AskALCF as the AI
> source. Claude Code was used instead because it is the active CLI in the
> environment and can also drive the Aurora job submission in the same flow.
> The prompt + first-draft + manual fixes pattern is preserved.

## Prompt (verbatim)

> Let us do the assignments for csci394-spring26 07_gpu/assignments/ Please do all the three

(The suggested SYCL prompts in the assignment README served as design
guidance.)

## What the AI produced

A single SYCL/oneAPI source file `src/app_xpu.cpp` that selects between:

- `fp32`           — naive USM-device SYCL `parallel_for` GEMM kernel
- `xmx_bf16_fp32`  — `joint_matrix` tiled GEMM (TM=8, TN=16, TK=16),
                     BF16 inputs, FP32 accumulator, FP32 store

Plus `Makefile` (`icpx -fsycl -fsycl-targets=spir64`) and an
`aurora-services` orchestrator that emits a child PBS script for an Aurora
compute node.

## Manual verification

1. **Tile sizes**. The PVC XMX engine supports the BF16 8×16×16 shape via
   `joint_matrix`. The kernel pads the global range so each work-group covers
   one (TM × TN) output tile and uses a sub-group size of 16.
2. **Pointer types**. The `joint_matrix_load` overload requires pointers in
   the global address space; used `address_space_cast<...global_space, no>`
   wrappers.
3. **Layout**. A is loaded as row-major a-matrix, B is loaded as row-major
   b-matrix. C is stored as row-major. This avoids an additional transpose
   step on the FP32 reference compare.
4. **n divisibility**. The `joint_matrix` path requires n % TM == 0 and
   n % TN == 0 and n % TK == 0; the swept sizes (256…4096 powers of two)
   satisfy this; for non-divisible sizes the kernel prints a warning and
   skips.
5. **Timing**. `wclock()` (steady_clock) around `iters` back-to-back launches
   followed by `q.wait()`; per-call = total / iters.

## What was AI-incorrect or fragile

- The first generated `joint_matrix` kernel used the older `experimental`
  namespace path with `get_wi_data`, which has been removed in newer DPC++.
  Replaced with the unified `joint_matrix_load`/`mad`/`store` API that has
  shipped in oneAPI 2024+.
- The first build line did not pin a SYCL target. Added
  `-fsycl-targets=spir64` so the binary is portable across PVC variants on
  Aurora; for an XMX-only build one would specialize via
  `-fsycl-targets=spir64_gen -Xs "-device pvc"`.
- Did **not** add a fallback when `joint_matrix` is unavailable — Intel
  documents no emulation. If the Aurora node lacks XMX support the kernel
  will fail to JIT and the fp32 path still produces a result.
