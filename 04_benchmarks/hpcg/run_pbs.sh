#!/bin/bash
#PBS -N hpcg
#PBS -l select=2:ncpus=32:mpiprocs=32
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -A <account>
#PBS -l filesystems=eagle:home

set -euo pipefail

cd "${PBS_O_WORKDIR}"

# Example module setup; adjust for your cluster
# module purge
# module load hpcg
# module load mpi

# Path to HPCG binary (xhpcg is common on clusters)
HPCG_BIN=${HPCG_BIN:-xhpcg}

# Ensure hpcg.dat is in the current directory
if [ ! -f hpcg.dat ]; then
  echo "hpcg.dat not found in $(pwd)" >&2
  exit 1
fi

mpirun -np 64 "$HPCG_BIN"
