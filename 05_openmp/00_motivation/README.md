# OpenMP motivation

This example shows why OpenMP is useful by comparing serial vs parallel
execution on the same loop.

## What the code computes
It sums the series:
```
sum = Σ (1 / i^2), for i = 1..N
```
This loop is independent across iterations, so it is safe to parallelize
with a reduction. The sum converges to ~1.644934 (π²/6) as N grows.

## Build
```bash
make
```

## Run
```bash
# Use more threads to see a speedup
OMP_NUM_THREADS=4 ./motivation
```

You can change the workload:
```bash
OMP_NUM_THREADS=8 ./motivation 400000000
```

## What to notice
- The same loop is run twice: once serial, once with `reduction`.
- The output prints wall-clock time and speedup.
