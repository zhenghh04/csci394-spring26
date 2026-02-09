# OpenMP set threads + manual self-distribution

This example sets the number of threads inside the program using
`omp_set_num_threads()` and then manually divides loop work among threads using
only `#pragma omp parallel` (no `#pragma omp for`).

## Build
```bash
make
```

## Run
```bash
./set_threads
```

## What to notice
- The program requests 4 threads using `omp_set_num_threads(4)`.
- Each thread computes its own `[start, end)` range from `thread_id` and
  `num_threads`.
- The code shows which thread handled which chunk and prints a test output.
