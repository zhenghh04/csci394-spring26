#!/bin/bash
#PBS -A DLIO
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q R7115122

cd ${PBS_O_WORKDIR}
module use /soft/modulefiles
module load nvhpc/23.3

make clean && make

echo "=== Matrix-Matrix Multiplication: CUDA Performance on A100 ==="
./matmul
