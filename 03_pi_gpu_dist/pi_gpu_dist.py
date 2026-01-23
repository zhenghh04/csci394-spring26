#!/usr/bin/env python3
"""
Estimate pi with Monte Carlo using PyTorch + torch.distributed (MPI).

Run with:
  mpiexec -n 4 python3 pi_torch_mpi_dist.py --samples 100000000
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Monte Carlo pi estimate (PyTorch + MPI)")
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000_000,
        help="Total number of random samples across all ranks",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5_000_000,
        help="Samples per batch on each rank",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    return parser.parse_args()


def select_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % device_count)
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        torch.xpu.set_device(rank % device_count)
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_backend(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    if device.type == "xpu":
        return "ccl"
    return "gloo"


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise SystemExit("batch must be > 0")

    device = select_device(0)
    backend = select_backend(device)
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()

    device = select_device(rank)
    backend = select_backend(device)
    torch.manual_seed(args.seed + rank)

    base = args.samples // world
    extra = args.samples % world
    local_samples = base + (1 if rank < extra else 0)

    remaining = local_samples
    local_hits = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    while remaining > 0:
        n = min(args.batch, remaining)
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        local_hits += torch.sum(x * x + y * y <= 1.0).item()
        remaining -= n

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    end = time.perf_counter()

    hits_tensor = torch.tensor(local_hits, dtype=torch.int64)
    samples_tensor = torch.tensor(local_samples, dtype=torch.int64)
    dist.reduce(hits_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    elapsed_tensor = torch.tensor(end - start, dtype=torch.float64)
    dist.reduce(elapsed_tensor, dst=0, op=dist.ReduceOp.MAX)

    if rank == 0:
        total_hits = int(hits_tensor.item())
        total_samples = int(samples_tensor.item())
        pi_est = 4.0 * total_hits / total_samples
        device_name = device.type
        print(
            f"backend={backend} device={device_name} procs={world} "
            f"samples={total_samples} hits={total_hits} pi={pi_est:.8f} "
            f"time_s={elapsed_tensor.item():.6f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
