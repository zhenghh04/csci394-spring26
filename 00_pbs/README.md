# Running jobs on HPC Systems

This is a short tutorial for submitting and monitoring jobs with PBS Pro.

## Login and data transfer
These steps are written for someone who has never used a supercomputer before.

### 0) Log in to the system
```bash
ssh <your_netid>@<login-hostname>
```
After you log in, you should be on a **login node** (not a compute node).

### 0.1) Move files to the cluster 
From your laptop:
```bash
scp -r /path/to/directory/csci394-spring26 <your_netid>@<login-hostname>:/path/on/cluster
```
You can also directly do ``git clone`` from the login node

### 0.2) Go to your work directory
```bash
cd /path/on/cluster/csci394-spring26/04_benchmarks
```

### 0.3) Load modules
Module names vary by site. Start by listing what’s available:
```bash
module avail
```
Then load what you need (example):
```bash
module load conda
```

## Submit and run a job
This walkthrough uses the `mpi_pingpong/` example, but the steps apply to any benchmark in this folder.

### 1) Build the program
```bash
cd csci394-spring26/04_benchmarks/mpi_pingpong
make
```

### 2) Create a PBS script

PBS Pro is the scheduler that decides **when** and **where** your job runs. You do not log in to compute nodes directly; instead, you submit a job and PBS places it in a queue, allocates nodes, starts your script, and tracks its status. 

We use a PBS script to define the PBS job. A PBS script is the job submission file that tells PBS Pro **what to run** and **what resources to allocate** (nodes, cores, walltime, queue, account). When you run qsub, PBS reads the script, schedules your job, and executes the commands inside it on the allocated compute nodes.

Key ideas:
- Your job runs only on resources granted by PBS.
- The `#PBS` lines in a script tell PBS what resources to allocate.
- Jobs can wait in a queue until resources are available.

Use `run_pbs.sh` as a starting point. A minimal version looks like:
```bash
#!/bin/bash
#PBS -N pingpong                 # job name
#PBS -l select=2                 # 1 node
#PBS -l walltime=00:05:00         # max runtime (hh:mm:ss)
#PBS -A DLIO                      # allocation/project name
#PBS -j oe                        # join stdout and stderr
#PBS -q workq

cd "$PBS_O_WORKDIR"

# launching the program on compute nodes via mpiexec
mpiexec -n 2 --ppn 1 ./pingpong
```

**PBS options used above**
- `-N` sets the job name shown in `qstat`.
- `-A` specifies the allocation/project account to charge.
- `-q` selects a queue (e.g., `debug`, `workq`).
- `-l` requests resources (nodes, cores, walltime, filesystems, etc.).

**PBS working directory**
`PBS_O_WORKDIR` is an environment variable set by PBS that points to the
directory where you ran `qsub`. Most scripts `cd "$PBS_O_WORKDIR"` so the job
runs in the same folder where you submitted it.

**Environment variables you can access inside the submission script**
- `PBS_JOBID` job ID
- `PBS_O_WORKDIR` directory where you submitted the job
- `PBS_NODEFILE` list of allocated nodes
- `PBS_QUEUE` queue name

**Count allocated nodes (inside script)**
```bash
NUM_NODES=$(cat "$PBS_NODEFILE" | uniq | wc -l)
echo "Nodes allocated: $NUM_NODES"
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

Example output:
```text
(miniconda3)[hzheng@crux-uan-0001 ~]$ qstat
Job id                 Name             User              Time Use S Queue
---------------------  ---------------- ----------------  -------- - -----
179455.crux-pbs-0001   testjob          richp                    0 H workq
190128.crux-pbs-0001   wrf              gsever1           00:00:10 R workq
190129.crux-pbs-0001   wrf              gsever1                  0 H workq
190130.crux-pbs-0001   wrf              gsever1                  0 H workq
190216.crux-pbs-0001   gmvspafg_spalit* yomori            2211:52* R preemptable
190217.crux-pbs-0001   sqepp_spalitett* yomori            2342:58* R preemptable
190219.crux-pbs-0001   muselens_spalit* yomori            2477:57* R preemptable
190220.crux-pbs-0001   gmvspafg_spalit* yomori            2184:25* R preemptable
190221.crux-pbs-0001   sqepp_spalitett* yomori            2326:54* R preemptable
190222.crux-pbs-0001   muselens_spalit* yomori            2467:59* R preemptable
190317.crux-pbs-0001   NAMD_chain       anijhawan         00:00:00 R workq
190318.crux-pbs-0001   NAMD_chain       anijhawan         00:00:00 R workq
```

Job status codes in the `S` column:
- `Q` queued (waiting for resources)
- `R` running
- `H` held (paused by user or policy)
- `C` completed (finished; may disappear quickly)
- `E` exiting/ending

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

### 7) Transfer results back to your local computer
Run the following command on your local computer
```bash
scp user@crux.alcf.anl.gov:/path/to/folder/file /path_on_your_local_machine
```

## Running jobs interactively
Sometimes, for debugging purpose, one can also run jobs interactively, 

Step-by-step interactive run:
1. Request an interactive job (adjust queue/resources for your site):
```bash
qsub -I -l select=1 -l walltime=00:10:00 -q workq -A DLIO -l filesystems=home:eagle
```

2. Wait for the prompt to move to a compute node (you will see a new hostname such as ```x1000c0s0b0n0```). This new hostname is one of **compute nodes** get allocated (usually is the first one on the list).
```text
qsub -q workq -I -A datascience -l walltime=1:00:00 -l select=1 -l filesystems=eagle:grand:home
qsub: waiting for job 190321.crux-pbs-0001.head.cm.crux.alcf.anl.gov to start
qsub: job 190321.crux-pbs-0001.head.cm.crux.alcf.anl.gov ready

(miniconda3)[hzheng@x1000c0s0b0n0 ~]$ 
```

3. Confirm you are on a compute node and the job is active:
```bash
$ hostname
$ qstat -u $USER

crux-pbs-0001.head.cm.crux.alcf.anl.gov: 
                                                                 Req'd  Req'd   Elap
Job ID               Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
-------------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
190321.crux-pbs-000* hzheng   workq    STDIN      30806*   1 256    --  01:00 R 00:01
```


4. Go to your working directory and load modules if needed:
```bash
cd "$PBS_O_WORKDIR"
module load mpi
```
5. Run your program:
```bash
mpiexec -n 2 --ppn 1 ./pingpong
```
6. When finished, type `exit` to release the node.

## Common PBS commands
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

## MPI launch basics (mpiexec)
`mpiexec` is the launcher provided by your MPI implementation. It starts
multiple MPI ranks across the nodes allocated to your job and sets up the
communication environment so the ranks can talk to each other.

**What is a rank?** In MPI, a *rank* is just one running copy of your program.
If you launch 4 ranks, you are running 4 separate processes that can communicate
with each other. Ranks are numbered `0` to `N-1`.

```bash
# Run with N total ranks
mpiexec -n 4 ./my_program

# Run with ranks per node (common on Cray/ALCF systems)
mpiexec -n 64 --ppn 32 ./my_program

# Pin ranks to cores (example)
mpiexec -n 64 --ppn 32 --cpu-bind depth -d 1 ./my_program
```

**Common `mpiexec` options**
- `-n N` total number of MPI ranks (processes) to launch.
- `--ppn P` ranks per node (so total ranks = nodes * P).
- `--cpu-bind depth` bind each rank to a set of cores to reduce migration.
- `-d 1` (often used with `--cpu-bind depth`) sets the *core depth* to 1, meaning
  each rank is bound to 1 core. In general, **core depth** is the number of CPU
  cores assigned to each rank when using `--cpu-bind depth` (e.g., `-d 2` binds
  each rank to 2 cores).
 

## Tips
- Always `cd "$PBS_O_WORKDIR"` in batch scripts.
- Start with small jobs to validate scripts before scaling up.

## Glossary (quick terms)
- `PBS Pro` job scheduler used to submit and manage batch jobs.
- `login node` shared front-end where you edit files, compile, and submit jobs.
- `compute node` dedicated node where your batch job actually runs.
- `batch job` non-interactive job submitted to the scheduler.
- `interactive job` job that gives you a shell on a compute node.
- `queue` named pool of resources with its own limits and policies.
- `allocation` or `project` account that pays for compute time.
- `walltime` maximum time a job is allowed to run.
- `node` a single server in the cluster.
- `core` a CPU core on a node.
- `MPI rank` one MPI process (one running copy of your program) in a parallel job.
- `ppn` processes per node (often used with `mpiexec`).
- `module` environment module used to load compilers, MPI, etc.
