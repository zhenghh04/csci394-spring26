"""
Tensor-parallel matrix multiply: C = A * B with all-reduce.

Matrix dimensions:
    A : [M, K]
    B : [K, N]
    C : [M, N]  (the result)

Partitioning strategy (split the K dimension):
    Each rank holds a K/P-wide slice of A (columns) and B (rows).
    Every rank independently computes a partial product:

        local_C[rank] = A[:, k_start:k_end]  @  B[k_start:k_end, :]

    Each local_C has the full output shape [M, N] but only captures the
    contribution from its slice of the shared K dimension.  Summing all
    partial products across ranks with all-reduce reconstructs the exact
    result C = A @ B.

Communication pattern:
    all_reduce(SUM)  --  the only collective needed.

Run modes:
    python3 matmul_tensor_parallel.py                        # single process
    torchrun --standalone --nproc_per_node=2 matmul_tensor_parallel.py
    torchrun --standalone --nproc_per_node=4 matmul_tensor_parallel.py
"""

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
        local_rank = 0
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
    rank = dist.get_rank()    
    world_size = dist.get_world_size()
    return rank, world_size, device


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def matmul_tensor_parallel(A_full, B_full, rank, world_size, device):
    """
    Compute C = A @ B using tensor parallelism on the K dimension.

    Each rank owns columns A[:, k_start:k_end] and rows B[k_start:k_end, :].
    After each rank computes its partial product, all-reduce sums them.

    Returns:
        C           : [M, N] tensor with the full matrix product on every rank
        t_matmul_ms : GPU time for the local matmul (ms)
        t_allreduce_ms : GPU time for the all-reduce (ms, 0 if world_size==1)
    """
    M, K = A_full.shape
    K2, N = B_full.shape
    assert K == K2, "A columns must match B rows"
    assert K % world_size == 0, "K must be divisible by world_size"

    shard_size = K // world_size
    k_start = rank * shard_size
    k_end = k_start + shard_size

    A_shard = A_full[:, k_start:k_end].to(device)   # [M, K/P]
    B_shard = B_full[k_start:k_end, :].to(device)   # [K/P, N]

    use_cuda = device.type == "cuda"

    # ---- time the local matmul ----
    if use_cuda:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()

    local_C = A_shard @ B_shard                      # [M, N]

    if use_cuda:
        e1.record()
        torch.cuda.synchronize()
        t_matmul_ms = e0.elapsed_time(e1)
    else:
        t_matmul_ms = 0.0
    if rank==0:
        print(
            f"[rank {rank}] K-slice [{k_start}:{k_end}]  "
            f"A_shard {tuple(A_shard.shape)}  "
            f"B_shard {tuple(B_shard.shape)}  "
            f"local_C {tuple(local_C.shape)}  "
            f"matmul={t_matmul_ms:.1f} ms"
        )

    # ---- time the all-reduce ----
    t_allreduce_ms = 0.0
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
            t_allreduce_ms = ea.elapsed_time(eb)

    return local_C, t_matmul_ms, t_allreduce_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size, device = setup_distributed()

    torch.manual_seed(42)

    M = 16384   # output rows
    K = 16384   # shared (contraction) dimension — must be divisible by world_size
    N = 16384   # output columns

    # All ranks construct the same full matrices so we can verify correctness.
    # In a real system each rank would only hold its own shard.
    A_full = torch.randn(M, K)
    B_full = torch.randn(K, N)

    if rank == 0:
        print("=== Tensor Parallel Matrix Multiply: C = A * B ===")
        print(f"World size : {world_size}")
        print(f"Device     : {device}")
        print(f"A shape    : {tuple(A_full.shape)}  (M={M}, K={K})")
        print(f"B shape    : {tuple(B_full.shape)}  (K={K}, N={N})")
        print(f"C shape    : [{M}, {N}]")
        print(f"K per rank : {K // world_size}")
        print()

    if is_distributed():
        dist.barrier()

    # Warmup
    _ = torch.randn(128, 128, device=device) @ torch.randn(128, 128, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    C_parallel, t_matmul_ms, t_allreduce_ms = matmul_tensor_parallel(
        A_full, B_full, rank, world_size, device
    )
    elapsed_wall = (time.perf_counter() - t0) * 1000

    if is_distributed():
        dist.barrier()

    if rank == 0:
        C_reference = A_full @ B_full

        max_err = (C_parallel.cpu() - C_reference).abs().max().item()
        rel_err = max_err / C_reference.abs().max().item()
        print()
        print(f"Wall time (rank 0, end-to-end)       : {elapsed_wall:.1f} ms")
        print(f"GPU time  — local matmul             : {t_matmul_ms:.1f} ms")
        print(f"GPU time  — all-reduce               : {t_allreduce_ms:.1f} ms")
        print(f"Max absolute error vs reference      : {max_err:.2e}")
        print(f"Max relative error vs reference      : {rel_err:.2e}")
        assert rel_err < 1e-4, f"relative error {rel_err:.2e} too large"
        print("PASSED")

    cleanup_distributed()


if __name__ == "__main__":
    main()
