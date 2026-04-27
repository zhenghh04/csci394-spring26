"""
Minimal tensor-parallel demonstration with a single linear layer.

The script shows two common sharding patterns:

1. Column parallelism:
   split output features across ranks, then all-gather outputs
2. Row parallelism:
   split input features across ranks, then all-reduce partial sums

This is not a full LLM implementation. It is a classroom demo that makes the
communication pattern visible with a very small example.

Example commands:

    python3 tensor_parallel_linear_demo.py
    torchrun --standalone --nproc_per_node=2 tensor_parallel_linear_demo.py
    torchrun --standalone --nproc_per_node=4 tensor_parallel_linear_demo.py
"""

import os

import torch
import torch.distributed as dist


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 1, torch.device("cpu")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def column_parallel_demo(x, full_weight, full_bias, rank, world_size, device):
    out_features, in_features = full_weight.shape
    if out_features % world_size != 0:
        raise ValueError("out_features must be divisible by world_size")

    local_out = out_features // world_size
    start = rank * local_out
    stop = start + local_out

    weight_shard = full_weight[start:stop].to(device)
    bias_shard = full_bias[start:stop].to(device)
    local_output = x @ weight_shard.t() + bias_shard

    if world_size > 1:
        gathered = [torch.zeros_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered, local_output)
        combined_output = torch.cat(gathered, dim=-1)
    else:
        combined_output = local_output

    reference = x @ full_weight.to(device).t() + full_bias.to(device)
    max_error = (combined_output - reference).abs().max().item()

    print(
        f"[rank {rank}] column shard weight shape = {tuple(weight_shard.shape)}, "
        f"local output shape = {tuple(local_output.shape)}"
    )
    if rank == 0:
        print("\nColumn parallel result:")
        print(combined_output.cpu())
        print(f"Max error vs dense linear layer: {max_error:.6f}\n")


def row_parallel_demo(x, full_weight, full_bias, rank, world_size, device):
    out_features, in_features = full_weight.shape
    if in_features % world_size != 0:
        raise ValueError("in_features must be divisible by world_size")

    local_in = in_features // world_size
    start = rank * local_in
    stop = start + local_in

    x_shard = x[:, start:stop].to(device)
    weight_shard = full_weight[:, start:stop].to(device)
    partial_output = x_shard @ weight_shard.t()

    if world_size > 1:
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)

    combined_output = partial_output + full_bias.to(device)
    reference = x @ full_weight.to(device).t() + full_bias.to(device)
    max_error = (combined_output - reference).abs().max().item()

    print(
        f"[rank {rank}] row shard input shape = {tuple(x_shard.shape)}, "
        f"weight shape = {tuple(weight_shard.shape)}"
    )
    if rank == 0:
        print("Row parallel result:")
        print(combined_output.cpu())
        print(f"Max error vs dense linear layer: {max_error:.6f}")


def main():
    rank, world_size, device = setup_distributed()
    torch.manual_seed(0)

    batch_size = 2
    in_features = 8
    out_features = 6

    x = torch.arange(batch_size * in_features, dtype=torch.float32, device=device).reshape(batch_size, in_features)
    full_weight = torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features) / 10.0
    full_bias = torch.linspace(-0.5, 0.5, out_features, dtype=torch.float32)

    if rank == 0:
        print("=== Tensor Parallel Linear Demo ===")
        print(f"World size: {world_size}")
        print(f"Input shape: {tuple(x.shape)}")
        print(f"Weight shape: {tuple(full_weight.shape)}")
        print()

    column_parallel_demo(x, full_weight, full_bias, rank, world_size, device)
    if is_distributed():
        dist.barrier()
    row_parallel_demo(x, full_weight, full_bias, rank, world_size, device)

    cleanup_distributed()


if __name__ == "__main__":
    main()
