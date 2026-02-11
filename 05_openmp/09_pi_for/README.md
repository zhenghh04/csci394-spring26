# OpenMP Pi

This folder has:
- `pi_omp.c`: OpenMP Monte Carlo pi example.

## Build C example
```bash
make
```

## Run OpenMP pi
```bash
OMP_NUM_THREADS=1 ./pi_omp 100000000
OMP_NUM_THREADS=8 ./pi_omp 100000000
```
