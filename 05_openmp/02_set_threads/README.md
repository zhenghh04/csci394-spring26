# OpenMP set threads explicitly

This example sets the number of threads inside the program using
`omp_set_num_threads()` instead of `OMP_NUM_THREADS`.

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
- Actual threads may differ if the runtime caps threads or if nesting is enabled.
