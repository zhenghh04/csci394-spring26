# OpenMP parallel for

Three versions of the same array-addition loop:
- `parallel_for` (serial)
- `parallel_for_omp` (OpenMP `parallel for` + timing)
- `parallel_for_dist` (prints which thread handles each index)

## Build
```bash
make
```

## Run
```bash
./parallel_for
OMP_NUM_THREADS=4 ./parallel_for_omp
OMP_NUM_THREADS=4 ./parallel_for_dist
```

## What to notice
- The OpenMP version splits the loop across threads.
- Each iteration is independent, so it is safe to parallelize.
- `parallel_for_dist` shows how iterations are distributed.
