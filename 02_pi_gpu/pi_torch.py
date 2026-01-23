#!/usr/bin/env python3
"""
Estimate pi with Monte Carlo using PyTorch on CPU/CUDA/XPU/MPS.

Run with:
  python3 pi_torch_device.py --device auto --samples 100000000
"""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi estimate (PyTorch, selectable device)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "xpu", "mps"],
        help="Device to use: auto, cpu, cuda, xpu, mps",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000_000,
        help="Total number of random samples",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10_000_000,
        help="Samples per batch to control memory use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Check NVIDIA drivers and CUDA install.")
        return torch.device("cuda")

    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise SystemExit("Intel XPU is not available. Check PyTorch XPU install and drivers.")
        return torch.device("xpu")

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available. Check PyTorch MPS build and macOS version.")
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise SystemExit("batch must be > 0")

    device = select_device(args.device)
    torch.manual_seed(args.seed)

    remaining = args.samples
    hits = 0

    synchronize(device)
    start = time.perf_counter()
    while remaining > 0:
        n = min(args.batch, remaining)
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        hits += torch.sum(x * x + y * y <= 1.0).item()
        remaining -= n
    synchronize(device)
    end = time.perf_counter()

    pi_est = 4.0 * hits / args.samples
    print(
        f"device={device.type} samples={args.samples} hits={hits} pi={pi_est:.8f} "
        f"time_s={end - start:.6f} batch={args.batch}"
    )


if __name__ == "__main__":
    main()
