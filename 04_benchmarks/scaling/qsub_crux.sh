```bash
#!/bin/bash
#PBS -N mpi_pi_scaling
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -A DLIO
#PBS -q workq
#PBS -l filesystems=10
cd "$PBS_O_WORKDIR"

# Get out how many nodes are allocated
NUM_NODES=$(cat "$PBS_NODEFILE" | uniq | wc -l)

# Strong scaling

SAMPLES=100000000
for n in 1 2 4 8 16 32 64 128; do
  mpiexec -n $n python pi_mpi4py.py --samples $SAMPLES
done

# Weak scaling
SAMPLES_PER_RANK=10000000
for n in 1 2 4 8 16 32 64 128; do
  total=$((SAMPLES_PER_RANK * n))
  mpiexec -n $n python pi_mpi4py.py --samples $total
done