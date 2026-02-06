#!/bin/bash
#PBS -N hpl
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -A datascience
#PBS -l filesystems=home:eagle

set -euo pipefail

cd "${PBS_O_WORKDIR}"

# Path to HPL binary (xhpl is common on clusters)
HPL_BIN=${HPL_BIN:-"./hpl-2.3/build/bin/xhpl"}


export NUM_NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
export PPN=128

mpiexec -np $(($NUM_NODES * $PPN)) --ppn $PPN --cpu-bind depth -d 1 "$HPL_BIN" 