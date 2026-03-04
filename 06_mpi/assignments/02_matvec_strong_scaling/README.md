# MPI Assignment: Matrix-Vector Multiply Strong Scaling

## Objective
Implement `main_mpi.c` for dense matrix-vector multiply using MPI row domain decomposition:
- `MPI_Scatter` rows of `A`
- `MPI_Bcast` full vector `x`
- local compute `y_local = A_local * x`
- `MPI_Gather` full result vector `y` on root

Then measure strong scaling up to 16 nodes on **Crux** (`ppn=8`):
- `nproc = 8, 16, 32, 64, 128` with `ppn = 8` 

## Communication Pattern
1. Root owns full `A (n x n)` and `x (n)`.
2. Scatter equal row blocks of `A` to each rank.
3. Broadcast `x` to all ranks.
4. Each rank computes local rows of `y`.
5. Gather local `y` chunks back to root.

## Performance measurement
Please measure the timing
1. End-to-end timing which include the whole process of scatter, broadcast, compute, and gather
2. Timing for each individual part scatter, broadcast, compute, and gather
3. Plot end-to-end timing vs nproc, and explain where the scaling bottleneck is from. 

## Files in This Assignment
- `main_mpi.c`: file you implement for MPI matvec with timing output.
- `Makefile`: builds `main_mpi.x`.
- 

## Build
```bash
make
```

## Single Run
```bash
mpiexec -n 8 --ppn 8 ./main_mpi.x 128000 5 1
```

Arguments:
- `n`: matrix/vector size (`n x n` matrix, `n` vector)
- `iters`: measured iterations
- `warmup`: warmup iterations (excluded from measured timing)

Parameters to script:
1. `n` (please set ``128000`` for the  experiments)
2. `iters` (default `5`)
3. `warmup` (default `1`)

## Parallel Efficiency
For each process count `p`:
- `speedup(p) = T(1) / T(p)`
- `efficiency(p) = speedup(p) / p`

where `T(p)` is average measured runtime from repeats.

## Deliverables
1. Source code: `main_mpi.c`, `Makefile`.
3. Timing table (average seconds over repeats), format:

| phase \\ nodes | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| scatter |  |  |  |  |  |
| broadcast |  |  |  |  |  |
| matvec_local |  |  |  |  |  |
| gather |  |  |  |  |  |
| total |  |  |  |  |  |

Use the same `n`, `iters`, and `warmup` for all node counts.
4. Scaling plot (for end to end): `scaling_plot.png`.
