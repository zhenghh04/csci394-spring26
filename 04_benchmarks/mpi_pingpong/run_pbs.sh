#!/bin/bash
#PBS -N mpi_pingpong   
#PBS -l select=4
#PBS -l walltime=00:05:00  
#PBS -j oe                 
#PBS -A DLIO               
#PBS -l filesystems=home:eagle

cd ${PBS_O_WORKDIR}  # cd to the directory where we submit the submission script

# Run (2 ranks on 2 nodes) - ensamble 
mpiexec -np 2 --ppn 1 ./pingpong --min 1 --max 64M --iters 1000 --warmup 100 
#mpiexec -np 2 --ppn 1 ./pingpong --min 1 --max 64M --iters 1000 --warmup 100 &
#wait 

