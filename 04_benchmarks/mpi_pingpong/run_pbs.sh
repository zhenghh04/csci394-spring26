#!/bin/bash
#PBS -N mpi_pingpong
#PBS -l select=2
#PBS -l walltime=00:05:00
#PBS -j oe
#PBS -A datascience
#PBS -l filesystems=home:eagle

set -euo pipefail

cd ${PBS_O_WORKDIR}

# Run (2 ranks on 2 nodes)
mpiexec -np 2 --ppn 1 ./pingpong --min 1 --max 64M --iters 1000 --warmup 100
