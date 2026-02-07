# Benchmarks

This folder contains small benchmarks you can run on a supercomputer to measure
node and filesystem performance. Each subfolder includes source code, a simple
Makefile, and a sample PBS script.

## Contents
- `hpl/` - HPL (Linpack) run template + example `HPL.dat`.
- `hpcg/` - HPCG run template + example `hpcg.dat`.
- `io/` - File-per-rank I/O bandwidth test.
- `mpi_pingpong/` - Two-rank ping-pong bandwidth/latency test.
- `scaling/` - Strong/weak scaling instructions using `pi_mpi4py.py`.

## Build quickstart
From any subfolder that contains a `Makefile`:
```bash
make
```

## Run quickstart
Use the provided `run_pbs.sh` as a starting point. Edit for your cluster's
module names, partitions, and MPI launcher.

For a step-by-step PBS workflow, see `PBS_PRO.md`.
