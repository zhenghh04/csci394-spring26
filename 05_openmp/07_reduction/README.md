# OpenMP reduction

`reduction` is used when multiple threads contribute to one shared result (for example a sum).  
Without reduction, `sum += value` inside a parallel loop causes a data race because threads update the same variable at the same time.

OpenMP `reduction(+:sum)` fixes this by:
- giving each thread a private local `sum`
- combining all local sums at the end of the loop

This folder includes `reduction.c`, which demonstrates two cases on a simple sum:
- **without reduction**: parallel sum of `1..100` (race-prone, may be wrong)
- **with reduction**: parallel sum of `1..100` using `reduction(+:sum)` (correct)

Expected value for `1 + 2 + ... + 100` is `5050`.

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./reduction_timing
```

You should see:
- `expected: 5050`
- `with_reduction: 5050`
