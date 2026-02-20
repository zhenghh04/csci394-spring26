# 07 Nonblocking Exchange

Uses `MPI_Irecv` and `MPI_Isend` with `MPI_Waitall` for neighbor exchange.

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./app
```
