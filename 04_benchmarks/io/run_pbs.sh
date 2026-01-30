#!/bin/bash
#PBS -N io_test
#PBS -l select=2:ncpus=4:mpiprocs=4
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o io_${PBS_JOBID}.out

# Example module setup; adjust for your cluster
# module purge
# module load mpi

set -euo pipefail

cd "${PBS_O_WORKDIR}"

make

# Writes 256 MB per rank per iteration to ./io_out
mpirun -np 8 ./io_test --mb 256 --iters 1 --chunk-kb 1024 --dir ./io_out
