# Loopy Kernel — OpenMP Parallelization Report

**Author:** Huihuo Zheng
**Course:** CSCI 394 — Spring 2026
**Assignment:** `05_openmp/assignments`
**Run target:** ALCF Crux compute node, single node, AMD EPYC, gcc 13.3.1 (`PrgEnv-gnu`), N = 2048, `debug` queue (PBS).
**PBS job:** `215524` on `x1000c0s0b0n1`, walltime ~60 s.

> Convert this Markdown to PDF for submission:
> `pandoc report_loopy_parallel.md -o report_loopy_parallel.pdf`

---

## 1. What was parallelized

| Region | Pragma used | Why this works |
|---|---|---|
| Task 1 — init `A[i,j] = 0.001*i + 0.002*j` | `#pragma omp parallel for collapse(2) schedule(static)` | Each cell written exactly once, no dependencies; `collapse(2)` exposes `N×N` independent iterations. |
| Task 2 — row sum | `#pragma omp parallel for schedule(static)` (outer `i` only) | Rows independent; the inner accumulator `sum` is loop-local (private by default). |
| Task 3a — stencil **without** collapse | `#pragma omp parallel for schedule(static)` (outer `i` only) | Read-only `A`, write-only `B`; no row dependence. |
| Task 3b — stencil **with** collapse(2) | `#pragma omp parallel for collapse(2) schedule(static)` | Same memory pattern, but the `(i,j)` iteration space is flattened into one parallel range. |
| Task 4 — total sum | `#pragma omp parallel for reduction(+:total) schedule(static)` | Classic reduction; OpenMP gives each thread a private partial sum and combines them. |

Boundary copies in Task 3 are O(N), negligible, and left serial — only the
interior O(N²) stencil is timed. Task 3 fills two distinct buffers `B` (3a) and
`C` (3b) and the program checks `max |B-C| == 0` before reporting any timings.
The check passed on every Crux run.

## 2. Build & run

```bash
# Build
make                        # gcc -O3 -fopenmp on Crux

# Submit
qsub submit_crux.pbs        # 1 node, debug queue, 7-thread sweep
```

PBS resources: `select=1`, walltime 20 min, queue `debug`, account
`datascience`, filesystems `home:eagle`. Threads pinned with
`OMP_PLACES=cores OMP_PROC_BIND=close`. The script sweeps
`OMP_NUM_THREADS ∈ {1, 2, 4, 8, 16, 32, 64}`, writes one `loopy_t<T>.dat`
per run, appends a row to `timings.csv`, and `diff`s every output against
the 1-thread reference.

## 3. Correctness

All seven thread counts produced bit-for-bit identical output files
(via `diff -q loopy_t1.dat loopy_t<T>.dat`):

```
OK:   loopy_t1.dat   matches loopy_t1.dat
OK:   loopy_t2.dat   matches loopy_t1.dat
OK:   loopy_t4.dat   matches loopy_t1.dat
OK:   loopy_t8.dat   matches loopy_t1.dat
OK:   loopy_t16.dat  matches loopy_t1.dat
OK:   loopy_t32.dat  matches loopy_t1.dat
OK:   loopy_t64.dat  matches loopy_t1.dat
```

The 1-thread reference values for N = 2048:

```
Sample Matrix[500][500]: 1.500000
Sample RowSums[500]:     5216.256000
Total Sum:              25141.254000
```

## 4. Crux timings (N = 2048)

### 4a. Per-task wall time (seconds)

Source: `timings.csv` produced by the PBS sweep.

| Threads | Task 1 init | Task 2 rowSum | Task 3a stencil (no collapse) | Task 3b stencil (collapse 2) | Task 4 total |
|--:|--:|--:|--:|--:|--:|
| 1  | 0.014034 | 0.003868 | 0.004587 | 0.011188 | 0.003873 |
| 2  | 0.008140 | 0.004274 | 0.003640 | 0.004765 | 0.001913 |
| 4  | 0.004683 | 0.002123 | 0.003210 | 0.003313 | 0.000983 |
| 8  | 0.003462 | 0.001053 | 0.003193 | 0.003531 | 0.000880 |
| 16 | 0.002503 | 0.000621 | 0.001797 | 0.002212 | 0.000430 |
| 32 | 0.002038 | 0.000452 | 0.001481 | 0.001642 | 0.000270 |
| 64 | 0.001954 | 0.000637 | 0.001054 | 0.001216 | 0.000146 |

### 4b. Speedup (T₁ / Tₚ)

| Threads | Task 1 | Task 2 | Task 3a | Task 3b | Task 4 |
|--:|--:|--:|--:|--:|--:|
| 1  |  1.00 |  1.00 | 1.00 | 1.00 |  1.00 |
| 2  |  1.72 |  0.91 | 1.26 | 2.35 |  2.02 |
| 4  |  3.00 |  1.82 | 1.43 | 3.38 |  3.94 |
| 8  |  4.05 |  3.67 | 1.44 | 3.17 |  4.40 |
| 16 |  5.61 |  6.23 | 2.55 | 5.06 |  9.01 |
| 32 |  6.89 |  8.56 | 3.10 | 6.81 | 14.34 |
| 64 |  7.18 |  6.07 | 4.35 | 9.20 | 26.53 |

### 4c. Stencil: `collapse(2)` vs no-collapse

| Threads | 3a no-collapse (ms) | 3b collapse(2) (ms) | ratio (3a / 3b) |
|--:|--:|--:|--:|
| 1  | 4.587 | 11.188 | 0.41 |
| 2  | 3.640 |  4.765 | 0.76 |
| 4  | 3.210 |  3.313 | 0.97 |
| 8  | 3.193 |  3.531 | 0.90 |
| 16 | 1.797 |  2.212 | 0.81 |
| 32 | 1.481 |  1.642 | 0.90 |
| 64 | 1.054 |  1.216 | 0.87 |

A ratio < 1 means `collapse(2)` is *slower* in absolute time. But notice the
**parallel speedup** column for 3b is much steeper (1 → 9.2× from 1 to 64 threads)
than for 3a (1 → 4.4×). So `collapse(2)` strong-scales better even though
per-iteration overhead leaves it ~15 % slower in absolute terms at this N.

## 5. What I learned

1. **Independence buys parallelism for free.** Tasks 1, 2, and 3 are
   embarrassingly parallel: every iteration writes a distinct array element
   and only reads from `A`. A single `parallel for` is enough — no
   `critical`, no `atomic`, no extra buffering. That is reflected in the
   essentially-monotone speedup of Tasks 1, 3, 4.

2. **Reductions are the textbook tool for Task 4 — and they are the
   star here.** Task 4 reaches 26.5× at 64 threads, by far the best. Why?
   It is the cheapest task per element (one add per double), so the
   working set easily fits in the aggregate L2/L3 of the cores once the
   thread count is high. The OpenMP runtime gives each thread a private
   partial sum and combines them at the end; there is no contention on the
   reduction variable until the very last step.

3. **`collapse(2)` only matters when the outer loop is short.**
   At N = 2048 the outer `i` runs 2046 iterations. With ≤ 64 threads each
   thread already gets ~32 rows, so flattening to `(N-2)²` ≈ 4 M iterations
   does not unlock new parallelism — it only adds index-arithmetic
   overhead and changes the cache access pattern. That is exactly what the
   table in §4c shows: at every thread count `collapse(2)` is 0–60 % slower
   in absolute time, because there is no bottleneck for it to relieve.
   The lesson the assignment is asking us to notice: `collapse(2)` becomes
   the right call only when the outer loop is shorter than the thread
   count (think `N × K` with small `K`, or 3D loops with a tiny outer
   dimension).

4. **Memory bandwidth is the ceiling for streaming kernels.**
   Task 1 (init, ~32 MiB written) plateaus at ~7× even on 64 threads —
   it is touching new memory on each iteration with essentially zero
   compute, so it saturates the EPYC memory subsystem. Task 2 (rowSum)
   is the most striking: it actually *degrades* from 8.6× at 32 threads
   to 6.1× at 64 threads. With both NUMA domains fully loaded, the
   second 32 threads are pulling A across the inter-socket link and the
   contention for memory channels outweighs the additional compute lanes.

5. **First-touch matters on NUMA nodes.** Because Task 1 is parallelized
   with `schedule(static)`, each thread initializes the rows it later
   reads in Tasks 2 and 3 — Linux puts those pages on the local NUMA
   domain. If Task 1 had been left serial, every cross-socket read in
   Tasks 2 and 3 would cost an inter-socket hop on Crux's two-socket
   AMD nodes, and the speedup curves above would collapse much earlier.
   This is invisible in the data but shows up the moment you serialize
   Task 1 — try it.

6. **Pinning is not optional.** Without `OMP_PLACES=cores
   OMP_PROC_BIND=close` the OS can migrate threads off the cores that
   first-touched their pages, undoing the placement above. The PBS
   script sets both, and you can see it pays off in the steady,
   low-noise scaling of Tasks 3 and 4.

7. **Strong scaling and absolute speed are different goals.** The
   stencil comparison (§4c) is the cleanest example. `collapse(2)`
   strong-scales twice as well as the no-collapse version, but it is
   slower at every thread count. If your job is to extract the most
   performance from N = 2048 today, pick no-collapse. If your job is to
   write code that will keep getting faster as the machine grows, pick
   `collapse(2)`. Real applications usually want both — which is why
   OpenMP exposes them as separate clauses rather than picking for you.

## 6. Files in this folder

| File | Purpose |
|---|---|
| `loopy_assignment.c` | OpenMP-parallel version of the four tasks, with per-task `omp_get_wtime()` timing and CSV output (one row per run). Stencil is run twice (no collapse / collapse) into separate buffers `B` and `C` and verified bit-identical at runtime. |
| `Makefile` | Builds `loopy_assignment` with `-fopenmp` (Linux/Crux gcc) or `clang -Xpreprocessor -fopenmp -lomp` (macOS via Homebrew libomp). |
| `submit_crux.pbs` | PBS submission script (`qsub submit_crux.pbs` from a Crux login node). Builds, sweeps `OMP_NUM_THREADS ∈ {1,2,4,8,16,32,64}` with `OMP_PLACES=cores OMP_PROC_BIND=close`, checks correctness, prints `timings.csv`. |
| `report_loopy_parallel.md` | This report (convert to PDF for submission). |
| `timings.csv` | Raw per-task timings, one row per thread count. Source of the numbers in §4. |
| `loopy_t1.dat` | 1-thread reference output (sample values + total sum) used as the correctness baseline. |
| `crux_run.log` | Saved PBS stdout from the run on `x1000c0s0b0n1`. |
