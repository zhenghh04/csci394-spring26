#!/bin/bash
#PBS -A DLIO
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug

cd ${PBS_O_WORKDIR}

make clean && make

echo "=== Matrix-Matrix Multiplication: CUDA Performance on A100 ==="
./matmul
