#!/bin/bash
#PBS -A DLIO
#PBS -l select=4:ncpus=256
#PBS -l walltime=00:10:00
#PBS -N mpi4py_pi_crux
#PBS -l filesystems=eagle:home
cd $HOME/csci394-spring26/01_mpi_pi/
source /eagle/datasets/soft/crux/miniconda3.sh
# Run jobs on single node
export SAMPLES=1000000000
for n in 1 2 4 8 16 32 64 128 256
do
    mpiexec -n $n --ppn $n --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples $SAMPLES
done
# run jobs on multiple nodes
mpiexec -n 256 --ppn 128 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples $SAMPLES
mpiexec -n 512 --ppn 128 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples $SAMPLES

