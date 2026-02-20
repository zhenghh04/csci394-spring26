#!/bin/bash
#PBS -N mpi_pi_scaling
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -A DLIO
#PBS -q workq
#PBS -l filesystems=home:eagle
cd "$PBS_O_WORKDIR"

# Get out how many nodes are allocated
NUM_NODES=$(cat "$PBS_NODEFILE" | uniq | wc -l)
source 
# Strong scaling

# Weak scaling
SAMPLES_PER_RANK=10000000
