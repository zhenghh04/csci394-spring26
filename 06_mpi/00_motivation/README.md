# 00 Motivation

This example parallelizes a summation (`sum 1/i^2`) across MPI ranks and reduces the partial sums.

## Build
```bash
make
```

## Run
```bash
mpiexec -n 1 ./app 100000000
mpiexec -n 4 ./app 100000000
```

Compare elapsed time as rank count increases.
