# OpenMP hello

Minimal OpenMP example: a parallel region where each thread prints its ID.

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./hello_omp
```

## What to notice
- `#pragma omp parallel` starts a parallel region.
- Each thread executes the same block.
- `omp_get_thread_num()` and `omp_get_num_threads()` report thread ID and count.
- The number of threads are set by environment variable ``OMP_NUM_THREADS``
