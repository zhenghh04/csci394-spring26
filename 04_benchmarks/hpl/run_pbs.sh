#!/bin/bash
#PBS -N hpl
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -A datascience
#PBS -l filesystems=home:eagle

set -euo pipefail

#cd "${PBS_O_WORKDIR}"

# Path to HPL binary (xhpl is common on clusters)
HPL_BIN=${HPL_BIN:-"hpl-2.3/build/bin/xhpl"}

# Ensure HPL.dat is in the current directory
if [ ! -f HPL.dat ]; then
  echo "HPL.dat not found in $(pwd)" >&2
  exit 1
fi

mpirun -np 64  "$HPL_BIN"
