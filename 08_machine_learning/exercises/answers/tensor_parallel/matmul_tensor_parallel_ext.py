"""
Extension of matmul_tensor_parallel.py:
  - Accept M, K, N from the command line (or --size for the symmetric case).
  - Repeat each timed step --repeats times and report the mean.
  - Optionally append one row per run to a CSV for sweep analysis.
  - Optionally skip the reference verification (--no-verify) when M*K*N is
    too large to materialise an extra full matmul on rank 0.

Run modes
---------

    # single process
    python3 matmul_tensor_parallel_ext.py --size 4096

    # 4 GPUs, sweep CSV
    torchrun --standalone --nproc_per_node=4 matmul_tensor_parallel_ext.py \
        --M 8192 --K 8192 --N 8192 --repeats 5 --csv results/sweep.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed setup helpers
# ---------------------------------------------------------------------------

def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        return 0, 1, device

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def matmul_tp_step(A_full, B_full, rank, world_size, device):
    M, K = A_full.shape
    K2, N = B_full.shape
    assert K == K2
    assert K % world_size == 0, f"K={K} not divisible by world_size={world_size}"

    shard = K // world_size
    k0 = rank * shard
    k1 = k0 + shard
    A_shard = A_full[:, k0:k1].to(device)
    B_shard = B_full[k0:k1, :].to(device)

    use_cuda = device.type == "cuda"

    if use_cuda:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
    local_C = A_shard @ B_shard
    if use_cuda:
        e1.record()
        torch.cuda.synchronize()
        t_matmul = e0.elapsed_time(e1)
    else:
        t_matmul = 0.0

    t_allreduce = 0.0
    if world_size > 1:
        if use_cuda:
            ea = torch.cuda.Event(enable_timing=True)
            eb = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            ea.record()
        dist.all_reduce(local_C, op=dist.ReduceOp.SUM)
        if use_cuda:
            eb.record()
            torch.cuda.synchronize()
            t_allreduce = ea.elapsed_time(eb)

    return local_C, t_matmul, t_allreduce


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--size", type=int, default=16384,
                   help="Symmetric size — used when --M/--K/--N omitted.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the reference A @ B check (saves rank-0 memory).")
    p.add_argument("--csv", type=str, default="",
                   help="If set, append one summary row per run to this CSV.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    M = args.M if args.M is not None else args.size
    K = args.K if args.K is not None else args.size
    N = args.N if args.N is not None else args.size

    rank, world_size, device = setup_distributed()
    torch.manual_seed(args.seed)

    A_full = torch.randn(M, K)
    B_full = torch.randn(K, N)

    if rank == 0:
        print("=== Tensor Parallel Matrix Multiply (extended) ===")
        print(f"World size : {world_size}")
        print(f"Device     : {device}")
        print(f"M, K, N    : {M}, {K}, {N}")
        print(f"K per rank : {K // world_size}")
        print(f"Repeats    : {args.repeats}")
        print()

    if is_distributed():
        dist.barrier()

    # warmup
    _ = torch.randn(128, 128, device=device) @ torch.randn(128, 128, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    matmul_times = []
    allreduce_times = []
    wall_times = []
    last_C = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        C_parallel, t_m, t_a = matmul_tp_step(A_full, B_full, rank, world_size, device)
        wall = (time.perf_counter() - t0) * 1000
        matmul_times.append(t_m)
        allreduce_times.append(t_a)
        wall_times.append(wall)
        last_C = C_parallel
        if is_distributed():
            dist.barrier()

    def mean(xs):
        return sum(xs) / len(xs)

    t_matmul_mean = mean(matmul_times)
    t_allreduce_mean = mean(allreduce_times)
    wall_mean = mean(wall_times)

    if rank == 0:
        print(f"Wall (mean over {args.repeats}) : {wall_mean:.2f} ms")
        print(f"Local matmul                   : {t_matmul_mean:.2f} ms")
        print(f"All-reduce                     : {t_allreduce_mean:.2f} ms")

        if not args.no_verify:
            C_ref = A_full @ B_full
            max_err = (last_C.cpu() - C_ref).abs().max().item()
            rel_err = max_err / C_ref.abs().max().item()
            print(f"Max abs error vs reference     : {max_err:.2e}")
            print(f"Max rel error vs reference     : {rel_err:.2e}")
            assert rel_err < 1e-4, f"relative error {rel_err:.2e} too large"
            print("PASSED")

        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
            write_header = not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "world_size", "M", "K", "N", "repeats",
                        "t_matmul_mean_ms", "t_allreduce_mean_ms",
                        "wall_mean_ms",
                        "t_matmul_min_ms", "t_allreduce_min_ms",
                        "ratio_allreduce_over_matmul",
                    ])
                w.writerow([
                    world_size, M, K, N, args.repeats,
                    f"{t_matmul_mean:.3f}", f"{t_allreduce_mean:.3f}",
                    f"{wall_mean:.3f}",
                    f"{min(matmul_times):.3f}", f"{min(allreduce_times):.3f}",
                    f"{(t_allreduce_mean / max(t_matmul_mean, 1e-9)):.4f}",
                ])
            print(f"Appended row to {args.csv}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
