# OpenMP pitfalls

Small examples that demonstrate common OpenMP mistakes and why they matter.

## hello_omp_wrong.c
This example introduces a **data race** by writing to shared variables inside a
parallel region. Each thread updates the same `tid` and `nthreads`, so the
results are non-deterministic.

Fixes:
- Declare `tid`/`nthreads` inside the parallel region, or
- Use `private(tid, nthreads)` on the parallel pragma.

## hello_omp_fixed.c
A corrected version that declares `tid` and `nthreads` inside the parallel region,
avoiding the race.
