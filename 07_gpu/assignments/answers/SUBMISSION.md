# GPU Assignments — Submission

This folder contains all three GPU group projects from
`07_gpu/assignments/README.md`. Each subfolder is a complete deliverable
(source, Makefile, run scripts, AI usage statement, report).

## Group / contributions

This submission was produced by a single agent run (Claude Opus 4.7) driving
the ALCF tooling on behalf of the user. Per the assignment rules, this is
documented honestly — see `AI_USAGE.md` in each project folder for prompt,
model, and what was changed manually.

## Layout

```text
01_gpu_offload_comparison/
  src/        — CPU / OpenACC / OpenMP target / CUDA / PyTorch GEMM
  Makefile    — Polaris (NVHPC + nvcc)
  AI_USAGE.md — prompt, model, manual fixes
  report.md   — 1–3 page report
  results/    — populated when the run completes
02_cublas_gemm_flops/
  src/        — cuBLAS FP32 + tensor-core (BF16->FP32) GEMM
  Makefile, AI_USAGE.md, report.md, results/
03_intel_xpu/
  src/        — SYCL FP32 + XMX joint_matrix GEMM
  Makefile, AI_USAGE.md, report.md, results/
```

## How the runs were dispatched

- **Project 01**: ClearML task `csci394_p1_offload_compare` →
  `polaris-services` queue → orchestrator on Polaris login node →
  `qsub` to PBS `debug` queue → 1× A100 compute node →
  build (NVHPC + nvcc + ALCF conda PyTorch) → sweep n=256..4096 → CSV + plot.
- **Project 02**: same pattern, ClearML task `csci394_p2_cublas_gemm_v3` →
  `polaris-services` → PBS `debug` queue → A100 → build (nvcc + cuBLAS) →
  sweep n=256..8192 → CSV + plot.
- **Project 03**: ClearML task `csci394_p3_xpu_gemm_v4` →
  `aurora-services` queue → orchestrator on Aurora login node →
  `qsub` to PBS `debug` queue → 1 PVC node → build (icpx -fsycl) → sweep
  n=256..4096 → CSV + plot.

Code/data movement: Globus from the local laptop endpoint
(`3ebfde51-41f4-11f1-9105-02535127e3d7`) to ALCF Eagle (Polaris) and
Flare (Aurora).

## Reproducing

Each project has a self-contained `Makefile` and `run.sh`. To reproduce on
your own ALCF allocation:

```bash
# Polaris (Project 01 / 02)
cd jobs/polaris/csci394_gpu_p1
qsub run.sh                # or use ClearML services queue

# Aurora (Project 03)
cd jobs/aurora/csci394_gpu_p3
qsub run.sh
```

The job folders are at:

- `jobs/polaris/csci394_gpu_p1/`
- `jobs/polaris/csci394_gpu_p2/`
- `jobs/aurora/csci394_gpu_p3/`

(see the user's main ALCF agentic_workflows repo)
