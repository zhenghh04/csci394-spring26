#!/bin/bash
#PBS -A DLIO
#PBS -l select=4:ncpus=256
#PBS -l walltime=00:10:00
#PBS -N mpi4py_pi_polaris
#PBS -l filesystems=eagle:home
#PBS -q debug-scaling
cd $HOME/csci394-spring26/01_mpi_pi/
module load conda
conda activate
# Run jobs on single node
export SAMPLES=100000000
for n in 1 2 4 8 16 32 64 
do
    mpiexec -n $n --ppn $n --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples $SAMPLES
done
# run jobs on multiple nodes
mpiexec -n 128 --ppn 64 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples $SAMPLES
mpiexec -n 256 --ppn 64 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples $SAMPLES

