# Assignment: MPI Pi Scaling Study on Crux

## Goal
Run **strong** and **weak** scaling tests for the MPI pi program on Crux using
**32 MPI ranks per node** and scaling up to **256 nodes**. You will collect
runtime data and compute speedup/efficiency for strong scaling.

## Program
Use the existing program (do not modify):
- `../../01_mpi_pi/pi_mpi4py.py`

## Required scale points
Use these node counts:
```
1, 2, 4, 8, 16, 32, 64, 128, 256
```
With 32 ranks per node, total ranks are:
```
32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
```

## Strong scaling (fixed global work)
Choose a single global sample count (e.g., `SAMPLES=1_000_000_000`) and keep it
constant for all node counts.

Command pattern:
```bash
mpiexec -n <ranks> --ppn 32 --cpu-bind depth -d 1 \
  python pi_mpi4py.py --samples <SAMPLES>
```

## Weak scaling (fixed work per rank)
Choose a per-rank sample count (e.g., `SAMPLES_PER_RANK=10_000_000`) and compute
`global_samples = samples_per_rank * total_ranks` for each run.

Command pattern:
```bash
total=$((SAMPLES_PER_RANK * ranks))
mpiexec -n <ranks> --ppn 32 --cpu-bind depth -d 1 \
  python pi_mpi4py.py --samples $total
```

## PBS batch template (Crux)
You can submit this once and it will run both studies:
```bash
#!/bin/bash
#PBS -A DLIO
#PBS -q debug-scaling
#PBS -l select=256:ncpus=32:mpiprocs=32
#PBS -l walltime=00:30:00
#PBS -N mpi_pi_scaling
#PBS -l filesystems=eagle:home
#PBS -j oe

cd $HOME/csci394-spring26/01_mpi_pi/
source /eagle/datasets/soft/crux/miniconda3.sh

# Strong scaling
SAMPLES=1000000000
for nodes in 1 2 4 8 16 32 64 128 256; do
  ranks=$((nodes * 32))
  mpiexec -n $ranks --ppn 32 --cpu-bind depth -d 1 \
    python pi_mpi4py.py --samples $SAMPLES
done

# Weak scaling
SAMPLES_PER_RANK=10000000
for nodes in 1 2 4 8 16 32 64 128 256; do
  ranks=$((nodes * 32))
  total=$((SAMPLES_PER_RANK * ranks))
  mpiexec -n $ranks --ppn 32 --cpu-bind depth -d 1 \
    python pi_mpi4py.py --samples $total
done
```

## What to submit
1. **Raw output log** (the file produced by PBS).
2. **Table** (CSV or Markdown) with columns:
   - `nodes`, `ranks`, `mode` (strong/weak), `samples`, `time_s`
3. **Strong-scaling metrics** for each node count:
   - speedup `S(p) = T1 / Tp`
   - efficiency `E(p) = S(p) / p`
4. **Two plots** (one for each scaling type):
   - Strong: `time_s` vs `ranks` (or speedup vs ranks)
   - Weak: `time_s` vs `ranks`

## Grading rubric (100 pts)
- 30 pts: correct strong-scaling runs (all node counts, fixed global samples)
- 30 pts: correct weak-scaling runs (all node counts, fixed per-rank samples)
- 20 pts: correct speedup/efficiency calculations
- 20 pts: clear plots + clean table

## Tips
- Keep `SAMPLES` high enough so runtime is measurable and not dominated by startup.
- If runs exceed time limits, reduce `SAMPLES` or `SAMPLES_PER_RANK` uniformly.
- Record the exact values you chose.
