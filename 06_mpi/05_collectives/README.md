# 05 Collectives

Separate demos for core MPI collectives:
- `bcast` -> `MPI_Bcast`
- `scatter` -> `MPI_Scatter`
- `gather` -> `MPI_Gather`
- `reduce` -> `MPI_Reduce`

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./bcast
mpiexec -n 4 ./scatter
mpiexec -n 4 ./gather
mpiexec -n 4 ./reduce
```
