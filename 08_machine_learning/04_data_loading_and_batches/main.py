#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install it with: python3 -m pip install torch") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data-loading and batch-size throughput example")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--samples", type=int, default=60000, help="Synthetic samples")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--height", type=int, default=28, help="Image height")
    parser.add_argument("--width", type=int, default=28, help="Image width")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the training loader")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.samples <= 0 or args.classes <= 1:
        raise SystemExit("Require batch-size > 0, samples > 0, and classes > 1")

    generator = torch.Generator().manual_seed(args.seed)
    images = torch.rand(args.samples, 1, args.height, args.width, generator=generator)
    labels = torch.randint(0, args.classes, (args.samples,), generator=generator)

    dataset = torch.utils.data.TensorDataset(images, labels)

    train_size = int(0.8 * args.samples)
    test_size = args.samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    start = time.perf_counter()
    train_batches = 0
    train_samples = 0
    label_sum = 0

    for batch_images, batch_labels in train_loader:
        train_batches += 1
        train_samples += batch_labels.size(0)
        label_sum += batch_labels.sum().item()
        _ = batch_images.mean().item()

    train_end = time.perf_counter()

    test_batches = 0
    test_samples = 0
    for batch_images, batch_labels in test_loader:
        test_batches += 1
        test_samples += batch_labels.size(0)
        _ = batch_images.std().item()

    end = time.perf_counter()

    train_time_s = train_end - start
    total_time_s = end - start
    train_throughput = train_samples / train_time_s if train_time_s > 0.0 else 0.0

    print("Data-loading and batch-size example")
    print(
        f"samples={args.samples} train_samples={train_samples} test_samples={test_samples} "
        f"batch_size={args.batch_size} workers={args.workers} shuffle={args.shuffle}"
    )
    print(
        f"train_batches={train_batches} test_batches={test_batches} "
        f"train_time_s={train_time_s:.6f} total_time_s={total_time_s:.6f} "
        f"train_samples_per_s={train_throughput:.2f} label_checksum={label_sum}"
    )


if __name__ == "__main__":
    main()
