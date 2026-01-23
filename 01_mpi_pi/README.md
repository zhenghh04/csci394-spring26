# MPI4PY Pi Example

This example estimates pi with a Monte Carlo method using `mpi4py`.

## Hands-on summary
- Understand parallelism with MPI ranks and work splitting.
- Run the same program with different process counts and measure scaling.
- Practice launching jobs on a supercomputer (PBS + `mpiexec`).
- Compare performance between a laptop and an HPC system.

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
mpiexec -n 1 python pi_mpi4py.py --samples 1000000

# More ranks: same total work, split across processes
mpiexec -n 4 python pi_mpi4py.py --samples 1000000
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

## GPU versions (NVIDIA + Intel XPU)
GPU versions live in `../02_pi_gpu` (no MPI required).

PyTorch (single-device, auto-select):
```bash
python -m pip install torch torchvision torchaudio
python3 ../02_pi_gpu/pi_torch.py --device auto --samples 100000000
```

Notes:
- NVIDIA version requires CUDA drivers.
- Intel XPU version requires PyTorch XPU build and oneAPI runtime.

## Run mpi4py example on your laptop
```bash
mpiexec -n 1 python pi_mpi4py.py --samples 100000000
mpiexec -n 2 python pi_mpi4py.py --samples 100000000
mpiexec -n 4 python pi_mpi4py.py --samples 100000000
mpiexec -n 8 python pi_mpi4py.py --samples 100000000
mpiexec -n 16 python pi_mpi4py.py --samples 100000000
```
### Notes
- `--samples` is the total number of random points across all ranks.
- Each rank uses a different random seed for independent sampling.
- `time_s` reports the max elapsed time across ranks for the compute+reduce phase.


**Example output on MacBook Pro M1 Max**

```bash
mpiexec -n 1 python pi_mpi4py.py --samples 100000000
procs=1 samples=100000000 hits=78536872 pi≈3.14147488 time_s=9.136346

mpiexec -n 2 python pi_mpi4py.py --samples 100000000
procs=2 samples=100000000 hits=78539839 pi≈3.14159356 time_s=4.553073

mpiexec -n 4 python pi_mpi4py.py --samples 100000000
procs=4 samples=100000000 hits=78546745 pi≈3.14186980 time_s=2.311936

mpiexec -n 8 python pi_mpi4py.py --samples 100000000
procs=8 samples=100000000 hits=78546039 pi≈3.14184156 time_s=1.176690

mpiexec -n 16 python pi_mpi4py.py --samples 100000000
procs=16 samples=100000000 hits=78541643 pi≈3.14166572 time_s=1.377759

mpiexec -n 32 python pi_mpi4py.py --samples 100000000
procs=32 samples=100000000 hits=78538842 pi≈3.14155368 time_s=1.430259
```

## Run on ALCF Crux (CPU only system)
**Crux hardware summary (ALCF)**
- Peak performance: 1.18 petaflops.
- Platform: HPE Cray EX with HPE Slingshot 11 interconnect (200 Gb).
- Compute node: 2x AMD EPYC 7742 (Rome) processors.
- User access nodes (UAN): 2x AMD EPYC 7543 (Milan) processors.
- System size: 256 compute nodes.

### Run interactively
```bash
# 1) Log in and load modules (names may vary)
ssh user@crux.alcf.anl.gov

# 2) clone the github repo
git clone https://github.com/zhenghh04/csci394-spring26.git

# 3) Interactive run (example)
qsub -I -l select=4:ncpus=256 -l walltime=1:00:00 -A DLIO -q debug -l filesystems=eagle:home

# 4) Loading environment
cd csci394-spring26/01_mpi_pi/
source /eagle/datasets/soft/crux/miniconda3.sh
# 5) Run jobs
export SAMPLES=100000000
mpiexec -n 1 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 2 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 4 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 8 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 16 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 32 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 64 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 256 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 256 --ppn 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
mpiexec -n 512 --ppn 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
```
### Run in batch mode
* Submission script: ``qsub_crux.sh``
    ```bash
    #!/bin/bash
    #PBS -A DLIO
    #PBS -l select=4:ncpus=256
    #PBS -l walltime=00:10:00
    #PBS -N mpi4py_pi_crux
    #PBS -l filesystems=eagle:home
    cd $HOME/csci394-spring26/01_mpi_pi/
    source /eagle/datasets/soft/crux/miniconda3.sh
    # Run jobs on single node
    export SAMPLES=100000000
    for n in 1 2 4 8 16 32 64 128 256
    do
        mpiexec -n $n --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
    done
    mpiexec -n 512 --ppn 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
    mpiexec -n 1024 --ppn 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
    ```
* Submitting job
    ```bash
    # 1) Log in and load modules (names may vary)
    ssh user@crux.alcf.anl.gov
    cd csci394-spring26/01_mpi_pi/
    qsub qsub_crux.sh
    # 2) Checking job status
    qstat -u $USER
    # 3) Checking results
    cat mpi4py_pi_crux.o*
    ```

### Results
```text
procs=1 samples=100000000 hits=78536872 pi=3.14147488 time_s=12.138612
procs=2 samples=100000000 hits=78539839 pi=3.14159356 time_s=6.061833
procs=4 samples=100000000 hits=78546745 pi=3.14186980 time_s=3.062778
procs=8 samples=100000000 hits=78546039 pi=3.14184156 time_s=1.543096
procs=16 samples=100000000 hits=78541643 pi=3.14166572 time_s=0.775980
procs=32 samples=100000000 hits=78538842 pi=3.14155368 time_s=0.394535
procs=64 samples=100000000 hits=78538686 pi=3.14154744 time_s=0.197432
procs=128 samples=100000000 hits=78541657 pi=3.14166628 time_s=0.101122
procs=256 samples=100000000 hits=78535779 pi=3.14143116 time_s=0.064277
procs=256 samples=100000000 hits=78535779 pi=3.14143116 time_s=0.068872
procs=512 samples=100000000 hits=78536864 pi=3.14147456 time_s=0.039481
```

## Run on ALCF Polaris
* Run interactively on a single node
```bash
# 1) Log in and load modules (names may vary)
ssh user@polaris.alcf.anl.gov

# 2) Interactive run (example)
qsub -I -l select=1:ncpus=256 -l walltime=00:10:00 -A DLIO -q debug -l filesystems=eagle:home

# 3) Loading environment and run the job
module load conda
conda activate
cd csci394-spring26/01_mpi_pi/
mpiexec -n 1 --cpu-bind depth -d 1 python pi_mpi4py.py --samples 100000000
mpiexec -n 2 --cpu-bind depth -d 1 python pi_mpi4py.py --samples 100000000
mpiexec -n 4 --cpu-bind depth -d 1 python pi_mpi4py.py --samples 100000000
mpiexec -n 8 --cpu-bind depth -d 1 python pi_mpi4py.py --samples 100000000
mpiexec -n 16 --cpu-bind depth -d 1 python pi_mpi4py.py --samples 100000000
mpiexec -n 32 --cpu-bind depth -d 1 python pi_mpi4py.py --samples 100000000
```

### Run on Batch mode: 
* Submission script: ``qsub_polaris.sh``
    ```bash
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
    for n in 1 2 4 8 16 32 64 128 256
    do
        mpiexec -n $n --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
    done
    mpiexec -n 256 --ppn 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
    mpiexec -n 512 --ppn 128 --cpu-bind depth -d 1 python pi_mpi4py.py --samples $SAMPLES
    ```

* Submitting job
    ```bash
    # 1) Log in and load modules (names may vary)
    ssh user@polaris.alcf.anl.gov
    cd csci394-spring26/01_mpi_pi/
    qsub qsub_polaris.sh
    # 2) Checking job status
    qstat -u $USER
    # 3) Checking results
    cat mpi4py_pi_polaris.o*
    ```

## Run on Aurora
```bash
ssh user@aurora.alcf.anl.gov
git clone https://github.com/zhenghh04/csci394-spring26.git
cd csci394-spring26/01_mpi_pi/
qsub qsub_aurora.sh
```    
## Comparing scaling performance 
![MPI4PY scaling plot (M1)](mpi4py_scaling_m1_vs_crux.png)

**System specs used in results**
- MacBook Pro M1 Max (CPU: 10 cores, 8 performance + 2 efficiency).
- ALCF Crux compute nodes: 2x AMD EPYC 7742 (64 cores each), 128 cores per node, 256 nodes total.
