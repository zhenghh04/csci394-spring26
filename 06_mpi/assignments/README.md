# MPI Assignments Overview

This folder contains three MPI assignments that build core performance-engineering skills:
1. Multi-rank communication bandwidth characterization.
2. Application-level strong scaling with matrix-vector multiply.
3. MPI+OpenMP hybrid parallelism.

## Assignments

## 1) Injection Bandwidth (Multi-Rank Half-to-Half Ping-Pong)
- Path: `01_injection_bandwidth/`
- Goal: Extend ping-pong to multi-rank half-to-half pairs and measure aggregate injection bandwidth on 2 nodes.
- Core MPI concepts:
- Rank pairing and point-to-point matching
- Warmup vs measured iterations
- Bottleneck timing via `MPI_Reduce(..., MPI_MAX, ...)`
- Aggregate bandwidth modeling
- Target systems: Crux, Aurora, Polaris.
- Main output: bandwidth-vs-total-ranks curves and saturation point per system.

## 2) Matrix-Vector Strong Scaling
- Path: `02_matvec_strong_scaling/`
- Goal: Implement dense matvec with row decomposition and study strong scaling.
- Core MPI concepts:
- `MPI_Scatter` rows of `A`
- `MPI_Bcast` vector `x`
- Local compute on row blocks
- `MPI_Gather` result `y`
- Main output: runtime/speedup/efficiency table and scaling plot.

## 3) MPI+OpenMP Hybrid Matvec Example
- Path: `03_mpi_openmp_hybrid/`
- Goal: Combine MPI (inter-process) and OpenMP (intra-process) for matvec.
- Core concepts:
- Hybrid decomposition (ranks x threads)
- Thread-level parallelism in local compute loop
- Tuning `OMP_NUM_THREADS` with fixed MPI ranks
- Main output: runtime comparison against MPI-only baseline.

## Combined Deliverables Checklist
1. Assignment 01 source + build files + 3 site scripts (Crux/Aurora/Polaris) + raw logs + report.
2. Assignment 02 source + build files + run script + timing table + scaling plot.
3. Assignment 03 hybrid source + build files + at least one run result.
4. Please add what you learned from the results.
5. Upload everything on Canvas.

## Notes
1. Exclude warmup iterations from measured performance metrics in both assignments.
2. Keep all experiment scripts and raw logs reproducible.
