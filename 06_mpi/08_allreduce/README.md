# 08 Allreduce

Computes a global L2 norm from per-rank local vectors using `MPI_Allreduce`.

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./app
# optional local vector length
mpiexec -n 4 ./app 200000
```
