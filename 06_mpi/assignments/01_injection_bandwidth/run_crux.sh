#!/bin/bash -l
set -e

echo "=== Assignment 1: Injection Bandwidth on Crux (2 nodes) ==="

# Build
cd /eagle/datascience/hzheng/csci394/06_mpi/assignments/01_injection_bandwidth
mpicc -O3 -std=c11 -Wall -Wextra -o app_injection main_injection.c

MSG_BYTES=1048576   # 1 MiB
ITERS=200
WARMUP=50

# Header
echo "system,nodes,total_ranks,message_bytes,bw_GBps,max_elapsed_s"

# Sweep even rank counts on 2 nodes (ppn goes from 1 to 8)
for TOTAL_RANKS in 2 4 8 12 16; do
    PPN=$((TOTAL_RANKS / 2))
    for REPEAT in 1 2 3; do
        echo "--- total_ranks=$TOTAL_RANKS ppn=$PPN repeat=$REPEAT ---"
        mpiexec -n $TOTAL_RANKS --ppn $PPN ./app_injection $MSG_BYTES $ITERS $WARMUP
    done
done

echo "=== Done ==="
