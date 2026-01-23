#!/usr/bin/env python3
"""
Estimate pi with Monte Carlo using PyTorch on Intel XPU.

Run with:
  python3 pi_torch_xpu.py --samples 100000000
"""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU Monte Carlo pi estimate (PyTorch XPU)")
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


def main() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit("Intel XPU is not available. Check PyTorch XPU install and drivers.")

    args = parse_args()
    if args.batch <= 0:
        raise SystemExit("batch must be > 0")

    device = torch.device("xpu")
    torch.manual_seed(args.seed)

    remaining = args.samples
    hits = 0

    torch.xpu.synchronize()
    start = time.perf_counter()
    while remaining > 0:
        n = min(args.batch, remaining)
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        hits += torch.sum(x * x + y * y <= 1.0).item()
        remaining -= n
    torch.xpu.synchronize()
    end = time.perf_counter()

    pi_est = 4.0 * hits / args.samples
    print(
        f"gpu=intel-xpu samples={args.samples} hits={hits} pi={pi_est:.8f} "
        f"time_s={end - start:.6f} batch={args.batch}"
    )


if __name__ == "__main__":
    main()
