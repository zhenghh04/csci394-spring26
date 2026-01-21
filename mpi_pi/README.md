# MPI4PY Pi Example

This example estimates pi with a Monte Carlo method using `mpi4py`.

![MPI4PY Pi schematic](mpi_pi_schematic.svg)

Each MPI rank generates random points, counts how many fall inside the unit circle,
and the counts are reduced to rank 0 to compute pi as `4 * (inside / total)`.

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
python3 -m pip install mpi4py
```

```powershell
# Windows (PowerShell)
py -m pip install mpi4py
```

## Run mpi4py example
```bash
mpiexec -n 4 python3 pi_mpi4py.py --samples 1000000
```

## Run on ALCF Polaris
```bash
# 1) Log in and load modules (names may vary)
ssh user@polaris.alcf.anl.gov

# 2) Interactive run (example)
qsub -I -l select=1:ncpus=4:mpiprocs=4 -l walltime=00:10:00 -A DLIO -q debug

# 3) Loading environment and run the job
module load conda
conda activate
mpiexec -n 1 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 2 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 4 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 8 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 16 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 32 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
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
mpiexec -n 1 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 2 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 4 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 8 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 16 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
mpiexec -n 32 --cpu-bind depth -d 2 python3 pi_mpi4py.py --samples 1000000
```

## Notes
- `--samples` is the total number of random points across all ranks.
- Each rank uses a different random seed for independent sampling.
- `time_s` reports the max elapsed time across ranks for the compute+reduce phase.
