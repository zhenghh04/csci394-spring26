#!/bin/bash -l
set -e

echo "=== Assignment 2: Matrix-Vector Strong Scaling on Crux ==="

# Build
cd /eagle/datascience/hzheng/csci394/06_mpi/assignments/02_matvec_strong_scaling
mpicc -O3 -std=c11 -Wall -Wextra -o main_mpi.x main_mpi.c
mpicc -O3 -std=c11 -Wall -Wextra -o main_series.x main_series.c

N=64000
ITERS=5
WARMUP=1

# Serial baseline
echo "--- Serial baseline ---"
./main_series.x $N $ITERS $WARMUP

# Strong scaling sweep: nproc = 8, 16, 32, 64, 128 with ppn=8
for NPROC in 8 16 32 64 128; do
    NODES=$((NPROC / 8))
    echo "--- nproc=$NPROC nodes=$NODES ppn=8 ---"
    mpiexec -n $NPROC --ppn 8 ./main_mpi.x $N $ITERS $WARMUP
done

echo "=== Done ==="
