# OpenMP `private` example

This example shows how `private` gives each thread its own copy of variables
inside a parallel region.

## Build
```bash
make
```

## Run
```bash
# Example: 4 threads
OMP_NUM_THREADS=4 ./private_example
```
