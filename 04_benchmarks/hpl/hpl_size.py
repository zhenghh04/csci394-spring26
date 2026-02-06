#!/usr/bin/env python3
"""Compute HPL problem size N from memory and node count."""

from __future__ import annotations

import argparse
import math
import re
import sys


_UNIT_TABLE = {
    "b": 1,
    "kb": 1000**1,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024**1,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*", value)
    if not match:
        raise ValueError(f"Invalid memory value: {value}")
    number = float(match.group(1))
    suffix = match.group(2).strip().lower()
    if suffix == "":
        suffix = "gb"
    if suffix not in _UNIT_TABLE:
        raise ValueError(
            "Unknown unit suffix. Use B, KB, MB, GB, TB, KiB, MiB, GiB, TiB."
        )
    return int(number * _UNIT_TABLE[suffix])


def clamp_positive_int(value: str) -> int:
    num = int(value)
    if num <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return num


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute HPL problem size N from memory and node count. "
            "Defaults assume --mem is per-node memory."
        )
    )
    parser.add_argument(
        "--mem",
        required=True,
        help="Memory per node (default units GB if no suffix, e.g. 256GB, 256GiB).",
    )
    parser.add_argument(
        "--num-nodes",
        required=True,
        type=clamp_positive_int,
        help="Number of nodes.",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.50,
        help="Fraction of total memory to use (default: 0.50).",
    )
    parser.add_argument(
        "--nb",
        type=clamp_positive_int,
        default=384,
        help="Round N down to a multiple of NB (default: 384).",
    )
    parser.add_argument(
        "--count",
        type=clamp_positive_int,
        default=1,
        help="How many N values to output (default: 1).",
    )
    parser.add_argument(
        "--step-percent",
        type=float,
        default=10.0,
        help="Percent step between sizes when --count > 1 (default: 10.0).",
    )
    parser.add_argument(
        "--total-mem",
        action="store_true",
        help="Treat --mem as total cluster memory instead of per-node.",
    )

    args = parser.parse_args()

    if not (0.0 < args.utilization <= 1.0):
        parser.error("--utilization must be in (0, 1].")

    mem_bytes = parse_bytes(args.mem)
    if not args.total_mem:
        mem_bytes *= args.num_nodes

    def compute_n(utilization: float) -> int:
        usable_bytes = utilization * mem_bytes
        n = int(math.sqrt(usable_bytes / 8.0))
        if n <= 0:
            parser.error("Computed N is non-positive; check inputs.")
        n -= n % args.nb
        if n <= 0:
            parser.error("Computed N is non-positive after NB rounding; check inputs.")
        return n

    values = []
    step = args.step_percent / 100.0
    for i in range(args.count):
        utilization = args.utilization - i * step
        if utilization <= 0.0:
            parser.error(
                "--count/--step-percent reduce utilization to <= 0; check inputs."
            )
        values.append(compute_n(utilization))

    print(" ".join(str(v) for v in reversed(values)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
