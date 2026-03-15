# 07 Nonblocking Exchange

Uses `MPI_Irecv` and `MPI_Isend` with `MPI_Waitall` for neighbor exchange.
Also includes a blocking deadlock demo in `main_block.c`.

## Schematic
![Nonblocking ring neighbor exchange](../assets/mpi_nonblocking_neighbor_exchange_schematic.svg)

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./app 1024
```

The command-line argument is the number of `int` values in the message.

Deadlock demo:
```bash
mpiexec -n 4 ./app_block 1024
```

`app_block` intentionally hangs because every rank performs `MPI_Ssend` before
posting the matching `MPI_Recv`.
The command-line argument is the number of `int` values in the message.
