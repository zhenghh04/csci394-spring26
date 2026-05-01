# 06_mpi — Answers

Submission for the two MPI assignments in `06_mpi/assignments/`.

## Layout

```text
answers/
├── 01_injection_bandwidth/
│   ├── main_injection.c              # half-to-half ping-pong with barriers
│   ├── Makefile                      # mpicc -O3 -std=c11
│   ├── submit_crux.pbs               # Crux PBS batch script
│   ├── submit_polaris.pbs            # Polaris PBS batch script
│   ├── submit_aurora.pbs             # Aurora PBS batch script
│   ├── results_crux.csv              # raw 21-row sweep, 3 repeats × 7 rank counts
│   ├── results_polaris.csv           # raw 18-row sweep, 3 repeats × 6 rank counts
│   ├── results_aurora.csv            # raw 24-row sweep, 3 repeats × 8 rank counts
│   ├── bandwidth_vs_ranks.png        # bandwidth curve per system
│   └── report_injection_bandwidth.md # full writeup (method, saturation, comparison)
└── 02_matvec_strong_scaling/
    ├── main_mpi.c                    # row-decomposed matvec with phase timers
    ├── main_series.c                 # serial baseline
    ├── Makefile
    ├── submit_crux.pbs               # Crux PBS batch script (workq-route, 16 nodes)
    ├── timing.csv                    # serial baseline + per-phase per-iter times
    ├── scaling_plot.png              # per-phase + speedup-vs-T(1)
    └── report_matvec_scaling.md      # full writeup with timing table and analysis
```

## Assignment 1 — Multi-rank injection bandwidth

Half-to-half ping-pong on 2 nodes (Crux, Polaris, Aurora), 1 MiB messages,
50 warmup + 200 measured iterations, 3 repeats per rank count.

**Peak measured (mean of 3 repeats, 2-node aggregate):**

| system | NICs/node | wire ceiling | peak measured | utilization |
|---|---:|---:|---:|---:|
| Crux | 1 × Slingshot 11 | 50 GB/s | 41.3 GB/s @ 128 ranks | 83% |
| Polaris | 2 × Slingshot 11 | 100 GB/s | 84.3 GB/s @ 64 ranks | 84% |
| Aurora | 8 × Slingshot 11 | 400 GB/s | 337.0 GB/s @ 208 ranks | 84% |

`wire ceiling = NICs/node × 25 GB/s/dir × 2 directions`. All three systems
sit near the same ~83–84% fraction of their NIC ceiling at peak; the
absolute number scales with the NIC count per node (1:2:8 ratio matches the
41:84:344 measured ratio).

See `01_injection_bandwidth/report_injection_bandwidth.md` for method,
formula, rank sweep per system, saturation criterion, and cross-system
comparison.

## Assignment 2 — Matrix-vector strong scaling (Crux)

`y = A·x` for `n = 64000`, `ppn = 8`, strong scaling on 1 → 16 nodes
(8 → 128 ranks). 1 warmup + 5 measured iterations per phase.

**Per-iteration timing:**

| phase \ nodes | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| scatter | 19.87 | 12.38 | 8.27 | 5.93 | 4.79 |
| broadcast | 2.19 | 2.60 | 2.75 | 1.41 | 0.71 |
| matvec_local | 0.89 | 0.47 | 0.26 | 0.12 | 0.06 |
| gather | 2.00 | 1.02 | 0.52 | 0.21 | 0.10 |
| **total** | **20.35** | **12.62** | **8.40** | **5.99** | **4.82** |

Serial baseline: T(1) = 3.785 s/iter. Speedup at 16 nodes / 128 ranks =
0.79× (still slower than serial). The matvec is **scatter-bound**, not
compute-bound — at 128 ranks the local compute is 0.06 s but `MPI_Scatter`
of A still costs 4.79 s, ≈99% of total time.

See `02_matvec_strong_scaling/report_matvec_scaling.md` for the full
analysis.

## Build and run (each assignment)

```bash
make
qsub submit_<system>.pbs
```

Output is appended to a CSV (`results_<system>.csv` for asn 1, `timing.csv`
for asn 2). Reports are markdown.
