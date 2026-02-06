# PBS Pro basics

This is a short cheat‑sheet for submitting and monitoring jobs with PBS Pro.

## Hands-on tutorial: submit and run a job
This walkthrough uses the `mpi_pingpong/` example, but the steps apply to any
benchmark in this folder.

### 1) Build the program
```bash
cd csci394-spring26/04_benchmarks/mpi_pingpong
make
```

### 2) Create a PBS script
Use `run_pbs.sh` as a starting point. A minimal version looks like:
```bash
#!/bin/bash
#PBS -N pingpong
#PBS -l select=1:ncpus=32:mpiprocs=2
#PBS -l walltime=00:05:00
#PBS -A DLIO
#PBS -j oe

cd "$PBS_O_WORKDIR"

# module load mpi   # adjust for your system
mpiexec -n 2 --ppn 1 ./pingpong
```

### 3) Submit the job
```bash
qsub run_pbs.sh
```
You will get a job ID like `123456.crux-pbs-0001`.

### 4) Monitor the job
```bash
qstat -u $USER
qstat -f 123456
```

### 5) View output
After the job finishes, check the output file:
```bash
ls -l
cat pingpong.o123456
```

### 6) Cancel or re-run
```bash
qdel 123456
qsub run_pbs.sh
```

### 7) Optional: interactive run
If your site allows it, request an interactive node:
```bash
qsub -I -l select=1 -l walltime=00:05:00 -q workq
```
Then run:
```bash
cd "$PBS_O_WORKDIR"
mpiexec -n 2 ./pingpong
```

## Common commands
```bash
qsub script.pbs        # submit a job
qstat -u $USER         # list your jobs
qstat -f <jobid>       # detailed job info
qdel <jobid>           # cancel a job
qhold <jobid>          # hold a job
qrls <jobid>           # release a held job
qstat -Q               # list queues
qstat -B               # server status
```

## Minimal PBS Pro script
```bash
#!/bin/bash
#PBS -N my_job
#PBS -l select=1:ncpus=32:mpiprocs=32
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o my_job.${PBS_JOBID}.out

cd "$PBS_O_WORKDIR"

# load modules (adjust for your system)
# module load conda
# conda activate

mpiexec -n 32 ./my_program
```

## MPI launch basics (mpiexec)
```bash
# Run with N total ranks
mpiexec -n 4 ./my_program

# Run with ranks per node (common on Cray/ALCF systems)
mpiexec -n 64 --ppn 32 ./my_program

# Pin ranks to cores (example)
mpiexec -n 64 --ppn 32 --cpu-bind depth -d 1 ./my_program
```

## Resource requests (common patterns)
```bash
# 2 nodes, 32 ranks per node
#PBS -l select=2:ncpus=32:mpiprocs=32

# Add a project/allocation (required at ALCF)
#PBS -A <account>

# Use a specific queue
#PBS -q debug

# Request filesystem access (required at ALCF)
#PBS -l filesystems=<fs1>:<fs2>
# Example (ALCF): #PBS -l filesystems=eagle:home
```

## Interactive jobs
```bash
qsub -I -l select=1:ncpus=32:mpiprocs=32 -l walltime=00:10:00 -q debug
```

## Environment variables you’ll see
- `PBS_JOBID` job ID
- `PBS_O_WORKDIR` directory where you submitted the job
- `PBS_NODEFILE` list of allocated nodes
- `PBS_QUEUE` queue name

## Count allocated nodes (inside script)
```bash
NUM_NODES=$(cat "$PBS_NODEFILE" | uniq | wc -l)
echo "Nodes allocated: $NUM_NODES"
```

## Tips
- Always `cd "$PBS_O_WORKDIR"` in batch scripts.
- Use `mpiexec --ppn <ranks_per_node>` if your site expects it.
- Start with small jobs to validate scripts before scaling up.
