#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install it with: python3 -m pip install torch") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed training starter with PyTorch DDP")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per process")
    parser.add_argument("--samples", type=int, default=8192, help="Synthetic samples")
    parser.add_argument("--input-dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit("Launch this program with torchrun, for example: torchrun --nproc_per_node=2 main.py")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def select_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_dataset(
    samples: int,
    input_dim: int,
    classes: int,
    seed: int,
) -> torch.utils.data.TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(samples, input_dim, generator=generator)
    weights = torch.randn(input_dim, classes, generator=generator)
    logits = features @ weights
    labels = torch.argmax(logits, dim=1)
    return torch.utils.data.TensorDataset(features, labels)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.samples <= 0:
        raise SystemExit("Require epochs > 0, batch-size > 0, and samples > 0")

    rank, world_size, local_rank = setup_distributed()
    torch.manual_seed(args.seed + rank)
    device = select_device(local_rank)

    dataset = build_dataset(args.samples, args.input_dim, args.classes, args.seed)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = torch.nn.Sequential(
        torch.nn.Linear(args.input_dim, args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, args.classes),
    ).to(device)

    if device.type == "cuda":
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    if rank == 0:
        print("Distributed training starter")
        print(
            f"world_size={world_size} device={device.type} samples={args.samples} "
            f"batch_size={args.batch_size} epochs={args.epochs}"
        )

    dist.barrier()
    synchronize(device)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        local_loss_sum = 0.0
        local_correct = 0
        local_seen = 0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = ddp_model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            local_loss_sum += loss.item() * labels.size(0)
            local_correct += (logits.argmax(dim=1) == labels).sum().item()
            local_seen += labels.size(0)

        metrics = torch.tensor(
            [local_loss_sum, float(local_correct), float(local_seen)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        if rank == 0:
            global_loss = metrics[0].item() / metrics[2].item()
            global_acc = metrics[1].item() / metrics[2].item()
            print(f"epoch={epoch} train_loss={global_loss:.6f} train_acc={global_acc:.4f}")

    dist.barrier()
    synchronize(device)
    end = time.perf_counter()

    if rank == 0:
        print(f"training_time_s={end - start:.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
