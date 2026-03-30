#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install it with: python3 -m pip install torch") from exc

try:
    from torchvision import datasets, transforms
except ImportError as exc:
    raise SystemExit(
        "torchvision is required for MNIST. Install it with: python3 -m pip install torchvision"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST linear classifier with PyTorch")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu", "mps"])
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional cap on training samples")
    parser.add_argument("--limit-test", type=int, default=0, help="Optional cap on test samples")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
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
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if requested == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise SystemExit("XPU is not available")
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise SystemExit("MPS is not available")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def maybe_limit(dataset: torch.utils.data.Dataset, limit: int) -> torch.utils.data.Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(limit)))


class LinearMNIST(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

    return total_loss / total_seen, total_correct / total_seen


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("Require epochs > 0 and batch-size > 0")

    torch.manual_seed(args.seed)
    device = select_device(args.device)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_dataset = maybe_limit(train_dataset, args.limit_train)
    test_dataset = maybe_limit(test_dataset, args.limit_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LinearMNIST().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    print("MNIST linear classifier")
    print(
        f"device={device.type} train_samples={len(train_dataset)} test_samples={len(test_dataset)} "
        f"batch_size={args.batch_size} epochs={args.epochs}"
    )

    synchronize(device)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.4f}"
        )

    synchronize(device)
    end = time.perf_counter()
    print(f"training_time_s={end - start:.6f}")


if __name__ == "__main__":
    main()
