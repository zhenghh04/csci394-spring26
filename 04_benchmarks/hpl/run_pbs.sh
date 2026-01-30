#!/bin/bash
#PBS -N hpl
#PBS -l select=2
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o hpl_${PBS_JOBID}.out
#PBS -A datascience
#PBS -l filesystems=home:eagle

# Example module setup; adjust for your cluster
# module purge
# module load hpl
# module load mpi

set -euo pipefail

cd "${PBS_O_WORKDIR}"

# Path to HPL binary (xhpl is common on clusters)
HPL_BIN=${HPL_BIN:-xhpl}

# Ensure HPL.dat is in the current directory
if [ ! -f HPL.dat ]; then
  echo "HPL.dat not found in $(pwd)" >&2
  exit 1
fi

mpiexec -np 64 --cpu-bind depth -d 1 "$HPL_BIN"
