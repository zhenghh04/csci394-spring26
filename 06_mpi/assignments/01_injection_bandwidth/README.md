# MPI Assignment: Multi-Rank Injection Bandwidth (Crux, Aurora, Polaris)

## Objective
Extend `../../04_ping_pong/main.c` from 2-rank ping-pong to a multi-rank half-to-half ping-pong benchmark and measure aggregate injection bandwidth on **2 nodes** on:
1. Crux
2. Aurora
3. Polaris

## Required Code Changes
Starting from `../../04_ping_pong/main.c`, implement a new program (for example `main_injection.c`) with these rules:
1. Use `MPI_COMM_WORLD` and require an even number of ranks.
2. Let `half = size / 2`.
3. Rank mapping:
   - for `rank < half`: partner is `rank + half` (send first, then recv)
   - for `rank >= half`: partner is `rank - half` (recv first, then send)
4. Keep command-line args: `message_bytes`, `iters`, `warmup`.
5. Run warmup iterations and **exclude warmup from measured statistics**.
6. Time only the measured loop using `MPI_Wtime()`.
7. Compute aggregate bandwidth using the bottleneck time:
   - `max_elapsed = max rank elapsed time` (via `MPI_Reduce(..., MPI_MAX, root=0)`)
   - `bw_GBps = (2 * message_bytes * iters * half) / max_elapsed / 1e9`
8. Print rank-0 summary in a parseable format.

## Communication Schematic

Use this half-to-half ping-pong pattern for your implementation:

![Half-to-half ping-pong schematic](../../04_ping_pong/ping_pong_pairs_schematic.svg)

## Build Requirement
Create your own build in this assignment folder 

Minimum requirement:
1. Build target for your injection benchmark executable.
2. Uses MPI compiler wrapper (`mpicc` or site equivalent).

## Experimental Plan (Must Run on 2 Nodes)
For **each system** (Crux, Aurora, Polaris):
1. Run with exactly 2 nodes.
2. Sweep total ranks upward (even values only):
   - start from a small even count (for example 2 or 4)
   - increase until aggregate bandwidth stops improving (saturation)
3. Keep message size fixed for saturation sweep (recommended: 1 MiB to 16 MiB).
4. Use warmup and measured iterations (example: `warmup=50`, `iters=200` or larger).
5. For each rank count, run at least 3 repeats and record average measured bandwidth.

## Saturation Criterion
Use and report a concrete criterion. Recommended:
- "Saturated" when bandwidth gain is < 5% for two consecutive rank increases.

## What to Submit
Submit one folder containing:
1. Source code:
   - your new benchmark source (for example `main_injection.c`)
   - your Makefile/CMake files
2. Submission scripts:
   - batch script for Crux
   - batch script for Aurora
   - batch script for Polaris
3. Results data:
   - raw run logs
   - a CSV table with at least these columns:
     - `system,nodes,total_ranks,message_bytes,bw_GBps,max_elapsed_s`
4. Report (`report_injection_bandwidth.pdf`, 1-3 pages) including:
   - benchmark method (including warmup exclusion)
   - rank sweep used per system
   - bandwidth-vs-ranks plot for each system 
   - identified saturation point per system
   - short comparison: Crux vs Aurora vs Polaris

## Suggested Commands (adapt to each site)
```bash
# build
make

# example run shape (2 nodes, even ranks)
mpiexec -n <TOTAL_RANKS> --ppn <RANKS_PER_NODE> ./app_injection <message_bytes> <iters> <warmup>
```

## Grading Focus
1. Correct partner mapping and communication logic.
2. Correct timing methodology (warmup excluded).
3. Proper 2-node experiments on all three systems.
4. Clear, reproducible scripts and reported evidence of saturation.

## Notes
1. Do not include warmup steps in performance statistics.
2. Keep logs and scripts reproducible so results can be rerun.
3. State any system limitations (queue limits, max ranks available, runtime caps).