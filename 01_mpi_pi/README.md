# MPI4PY Pi Example

This example estimates pi with a Monte Carlo method using `mpi4py`.

![MPI4PY Pi schematic](mpi_pi_schematic.svg)

Each MPI rank generates random points, counts how many fall inside the unit circle,
and the counts are reduced to rank 0 to compute pi as `4 * (inside / total)`.

## MPI quick start (no background needed)
- MPI runs the same program on many processes (called "ranks").
- `mpiexec -n <N>` launches `N` ranks; rank IDs go from `0` to `N-1`.
- In this example, each rank does part of the work, then MPI reduces results to rank 0.
- `--samples` is the total work across all ranks, not per rank.

Try these in order:
```bash
# One rank: behaves like a normal program
mpiexec -n 1 python3 pi_mpi4py.py --samples 1000000

# More ranks: same total work, split across processes
mpiexec -n 4 python3 pi_mpi4py.py --samples 1000000
```

## Requirements
- Python 3
- mpi4py
- An MPI runtime (Open MPI, MPICH, or similar)

Install Open MPI
```bash
# macOS (Homebrew)
brew install open-mpi

# Ubuntu/Debian
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev

# Fedora
sudo dnf install -y openmpi openmpi-devel
```

Windows (PowerShell, MS-MPI)
```powershell
# Install MS-MPI (run as Administrator)
winget install --id Microsoft.MPI

# Add the MS-MPI bin folder to PATH for this session
$env:Path += ";C:\Program Files\Microsoft MPI\Bin"
```

Install mpi4py
```bash
# macOS/Linux
python -m pip install --no-binary=mpi4py mpi4py
```

```powershell
# Windows (PowerShell)
py -m pip install mpi4py
```

## Run mpi4py example on your laptop
```bash
mpiexec -n 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 2 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 4 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 8 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 16 python3 pi_mpi4py.py --samples 100000000
```

## Run on ALCF Polaris
```bash
# 1) Log in and load modules (names may vary)
ssh user@polaris.alcf.anl.gov

# 2) Interactive run (example)
qsub -I -l select=1:ncpus=4:mpiprocs=4 -l walltime=00:10:00 -A DLIO -q debug -l filesystems=eagle:home

# 3) Loading environment and run the job
module load conda
conda activate
mpiexec -n 1 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 2 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 4 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 8 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 16 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 32 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
```

Batch job example:
```bash
#!/bin/bash
#PBS -A DLIO
#PBS -l select=1:ncpus=4:mpiprocs=4
#PBS -l walltime=00:10:00
#PBS -N mpi4py_pi

module load conda
conda activate mpi4py
mpiexec -n 1 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 2 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 4 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 8 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 16 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
mpiexec -n 32 --cpu-bind depth -d 1 python3 pi_mpi4py.py --samples 100000000
```

## Notes
- `--samples` is the total number of random points across all ranks.
- Each rank uses a different random seed for independent sampling.
- `time_s` reports the max elapsed time across ranks for the compute+reduce phase.

## Example results 

### MacBook Pro M1 Max

```text
mpiexec -n 1 python3 pi_mpi4py.py --samples 100000000
procs=1 samples=100000000 hits=78536872 pi≈3.14147488 time_s=9.136346

mpiexec -n 2 python3 pi_mpi4py.py --samples 100000000
procs=2 samples=100000000 hits=78539839 pi≈3.14159356 time_s=4.553073

mpiexec -n 4 python3 pi_mpi4py.py --samples 100000000
procs=4 samples=100000000 hits=78546745 pi≈3.14186980 time_s=2.311936

mpiexec -n 8 python3 pi_mpi4py.py --samples 100000000
procs=8 samples=100000000 hits=78546039 pi≈3.14184156 time_s=1.176690

mpiexec -n 16 python3 pi_mpi4py.py --samples 100000000
procs=16 samples=100000000 hits=78541643 pi≈3.14166572 time_s=1.377759

mpiexec -n 32 python3 pi_mpi4py.py --samples 100000000
procs=32 samples=100000000 hits=78538842 pi≈3.14155368 time_s=1.430259
```
### Crux at ALCF
```text
procs=1 samples=100000000 hits=78536872 pi≈3.14147488 time_s=14.763054
procs=2 samples=100000000 hits=78539839 pi≈3.14159356 time_s=7.327552
procs=4 samples=100000000 hits=78546745 pi≈3.14186980 time_s=3.717046
procs=8 samples=100000000 hits=78546039 pi≈3.14184156 time_s=1.846521
procs=16 samples=100000000 hits=78541643 pi≈3.14166572 time_s=0.928758
procs=32 samples=100000000 hits=78538842 pi≈3.14155368 time_s=0.465773
procs=64 samples=100000000 hits=78538686 pi≈3.14154744 time_s=0.273795
procs=128 samples=100000000 hits=78541657 pi≈3.14166628 time_s=0.143938
procs=256 samples=100000000 hits=78535779 pi≈3.14143116 time_s=0.147014
```
![MPI4PY scaling plot (M1)](mpi4py_scaling_m1_vs_crux.png)

**System specs used in results**
- MacBook Pro M1 Max (CPU: 10 cores, 8 performance + 2 efficiency).
- ALCF Crux compute nodes: 2x AMD EPYC 7742 (64 cores each), 128 cores per node, 256 nodes total.
