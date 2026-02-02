# MPI Pi scaling study

This folder uses the existing `pi_mpi4py.py` as-is.  
You run **strong** and **weak** scaling by changing only inputs (`-n` and
`--samples`), not the Python source.

## Program behavior reminder
- `--samples` is the **total global samples across all ranks**.
- Runtime line includes `procs=...` and `time_s=...`.

## Strong scaling
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

Use these formulas:
- speedup: `S(p) = T1 / Tp`
- efficiency: `E(p) = S(p) / p`

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

## PBS batch template
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
```
