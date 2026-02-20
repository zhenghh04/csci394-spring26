# MPI hands-on examples

This module mirrors the style of `05_openmp`: small, focused examples from basic to more practical MPI usage.

## Contents
- `00_motivation/` - why distributed work helps (distributed sum + `MPI_Reduce`).
- `01_hello/` - minimal MPI hello world with rank IDs.
- `02_rank_info/` - rank, world size, and hostname (`MPI_Get_processor_name`).
- `03_send_recv/` - first point-to-point message (`MPI_Send`/`MPI_Recv`).
- `04_ring/` - ring communication using `MPI_Sendrecv_replace`.
- `05_collectives/` - `Bcast`, `Scatter`, `Gather`, and `Reduce` in one example.
- `06_pi_reduce/` - numerical pi integration with distributed reduction.
- `07_nonblocking/` - neighbor exchange with `MPI_Isend`/`MPI_Irecv`.
- `08_allreduce/` - global L2 norm with `MPI_Allreduce`.

## Build
In any subfolder:
```bash
make
```

## Run
Use `mpiexec` (or `mpirun`) with rank count:
```bash
mpiexec -n 4 ./app
```

Examples:
```bash
cd 01_hello
make
mpiexec -n 4 ./app

cd ../06_pi_reduce
make
mpiexec -n 8 ./app 50000000
```

## Suggested order
1. `01_hello`
2. `02_rank_info`
3. `03_send_recv`
4. `04_ring`
5. `05_collectives`
6. `06_pi_reduce`
7. `07_nonblocking`
8. `08_allreduce`

## Notes
- Use one rank first (`-n 1`) to validate basics, then scale up.
- Many examples are easiest to inspect with `-n 2` or `-n 4`.
- On clusters, run these in interactive or batch jobs via PBS.
