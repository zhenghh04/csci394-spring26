#!/usr/bin/env python3
"""
Estimate pi with Monte Carlo using mpi4py.

Run with:
  mpiexec -n 1 python3 pi_mpi4py.py --samples 100000000
  mpiexec -n 2 python3 pi_mpi4py.py --samples 100000000
  mpiexec -n 4 python3 pi_mpi4py.py --samples 100000000
  mpiexec -n 8 python3 pi_mpi4py.py --samples 100000000
  ...
"""

from __future__ import annotations

import argparse
import random
from mpi4py import MPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI Monte Carlo pi estimate")
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000_000,
        help="Total number of random samples across all ranks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split the total work across ranks, distributing remainder to low ranks.
    base = args.samples // size
    extra = args.samples % size
    local_samples = base + (1 if rank < extra else 0)

    random.seed(42 + rank)

    # Time the compute+reduce phase for scaling studies.
    comm.Barrier()
    start = MPI.Wtime()

    local_hits = 0
    for _ in range(local_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            local_hits += 1

    total_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)
    total_samples = comm.reduce(local_samples, op=MPI.SUM, root=0)
    end = MPI.Wtime()

    max_elapsed = comm.reduce(end - start, op=MPI.MAX, root=0)

    if rank == 0:
        pi_est = 4.0 * total_hits / total_samples
        print(
            f"procs={size} samples={total_samples} hits={total_hits} "
            f"pi={pi_est:.8f} time_s={max_elapsed:.6f}"
        )

if __name__ == "__main__":
    main()
