# 07 Nonblocking Exchange

Uses `MPI_Irecv` and `MPI_Isend` with `MPI_Waitall` for neighbor exchange.

## Schematic
![Nonblocking ring neighbor exchange](../assets/mpi_nonblocking_neighbor_exchange_schematic.svg)

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./app
```
