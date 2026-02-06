# MPI Pi scaling study

This folder uses the existing `pi_mpi4py.py` as-is.  
You run **strong** and **weak** scaling by changing only inputs (`-n` and
`--samples`), not the Python source.

## Program behavior reminder
- `--samples` is the **total global samples across all ranks**.
- Runtime line includes `procs=...` and `time_s=...`.

## Strong scaling on a single node
Keep global work fixed; increase ranks.

```bash
export SAMPLES=100000000
for n in 1 2 4 8 16 32 64; do
  mpiexec -n $n python pi_mpi4py.py --samples $SAMPLES
done
```

Results on my laptop
```text
procs=1 samples=100000000 hits=78536872 pi=3.14147488 time_s=9.092849
procs=2 samples=100000000 hits=78539839 pi=3.14159356 time_s=4.548793
procs=4 samples=100000000 hits=78546745 pi=3.14186980 time_s=2.282205
procs=8 samples=100000000 hits=78546039 pi=3.14184156 time_s=1.159484
```

## Weak scaling
Keep work per rank fixed; increase ranks.  
Because the code expects global samples, set:

`global_samples = samples_per_rank * num_ranks`

```bash
export SAMPLES_PER_RANK=10000000
for n in 1 2 4 8 16 32 64; do
  total=$((SAMPLES_PER_RANK * n))
  mpiexec -n $n python pi_mpi4py.py --samples $total
done
```

In ideal weak scaling, `time_s` stays close to constant as `n` grows.

On my laptop
```test
procs=1 samples=10000000 hits=7855774 pi=3.14230960 time_s=0.914205
procs=2 samples=20000000 hits=15709806 pi=3.14196120 time_s=0.920944
procs=4 samples=40000000 hits=31417974 pi=3.14179740 time_s=0.913289
procs=8 samples=80000000 hits=62836626 pi=3.14183130 time_s=0.930731
procs=16 samples=160000000 hits=125665425 pi=3.14163563 time_s=2.017165
```

## Crux scaling assignment template (128 ranks/node)
Below is a Crux-oriented template that uses **128 MPI ranks per node** and scales
from 1 to 128 nodes. Adjust the account, queue, and filesystem settings for your
allocation.


```bash
#!/bin/bash
#PBS -A DLIO
#PBS -l select=64
#PBS -l walltime=00:30:00
#PBS -N mpi_pi_scaling
#PBS -l filesystems=eagle:home
#PBS -j oe

cd ${PBS_O_WORKDIR}

source /eagle/datasets/soft/crux/miniconda3.sh

NUM_NODES=$(cat "$PBS_NODEFILE" | uniq | wc -l)
PPN=128
RANKS=$((NUM_NODES * PPN))

# Strong scaling (fixed global work)
SAMPLES=1000000000
mpiexec -n $RANKS --ppn $PPN --cpu-bind depth -d 1 \
    python pi_mpi4py.py --samples $SAMPLES

# Weak scaling (fixed work per rank)
SAMPLES_PER_RANK=10000000
mpiexec -n $RANKS --ppn $PPN --cpu-bind depth -d 1 \
    python pi_mpi4py.py --samples $((SAMPLES_PER_RANK * RANKS))
```

Please run the scaling test from 1 node to 128 nodes. The strong-scaling section
keeps total work fixed; the weak-scaling section keeps work per rank fixed.

Deliverables:
1. Plot total time to solution vs. number of nodes for both cases.
2. Calculate scaling efficiency at 128 nodes:
   - Strong scaling: efficiency = (T1 / (T128 * 128)) * 100%.
   - Weak scaling: efficiency = (T1 / T128) * 100%.
