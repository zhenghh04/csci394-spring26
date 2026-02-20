# 06 Pi Reduce

Numerical integration of pi with distributed work and `MPI_Reduce`.

## Build
```bash
make
```

## Run
```bash
mpiexec -n 1 ./app 10000000
mpiexec -n 8 ./app 10000000
```
