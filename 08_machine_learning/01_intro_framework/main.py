#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install it with: python3 -m pip install torch") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intro PyTorch example: synthetic linear regression on a selectable device"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu", "mps"])
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--samples", type=int, default=1024, help="Synthetic training samples")
    parser.add_argument("--input-dim", type=int, default=16, help="Feature dimension")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Standard deviation of label noise")
    parser.add_argument("--lr", type=float, default=0.05, help="SGD learning rate")
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


def build_dataset(
    samples: int,
    input_dim: int,
    noise_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(samples, input_dim, generator=generator)
    true_weights = torch.linspace(-1.0, 1.0, input_dim, dtype=torch.float32).unsqueeze(1)
    true_bias = torch.tensor([0.25], dtype=torch.float32)
    noise = noise_std * torch.randn(samples, 1, generator=generator)
    targets = features @ true_weights + true_bias + noise
    return features, targets, true_weights, true_bias


def main() -> None:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.samples <= 0
        or args.input_dim <= 0
        or args.noise_std < 0.0
        or args.lr <= 0.0
    ):
        raise SystemExit(
            "Require epochs > 0, batch-size > 0, samples > 0, input-dim > 0, noise-std >= 0, and lr > 0"
        )

    torch.manual_seed(args.seed)
    device = select_device(args.device)

    features, targets, true_weights, true_bias = build_dataset(
        args.samples, args.input_dim, args.noise_std, args.seed
    )
    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = torch.nn.Linear(args.input_dim, 1).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print("Intro PyTorch linear regression")
    print(f"torch_version={torch.__version__}")
    print(
        f"device={device.type} samples={args.samples} batch_size={args.batch_size} "
        f"epochs={args.epochs} input_dim={args.input_dim} noise_std={args.noise_std:.3f} lr={args.lr:.3f}"
    )

    synchronize(device)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_seen = 0

        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_targets.size(0)
            total_seen += batch_targets.size(0)

        avg_loss = total_loss / total_seen
        print(f"epoch={epoch} mse={avg_loss:.6f}")

    synchronize(device)
    end = time.perf_counter()

    learned_weights = model.weight.detach().cpu().view(-1, 1)
    learned_bias = model.bias.detach().cpu().view(-1)
    weight_max_abs_err = torch.max(torch.abs(learned_weights - true_weights)).item()
    bias_abs_err = torch.abs(learned_bias - true_bias).item()

    with torch.no_grad():
        predictions = model(features.to(device)).cpu()
        fit_mse = torch.mean((predictions - targets) ** 2).item()

    print(f"training_time_s={end - start:.6f}")
    print(f"fit_mse={fit_mse:.6f} weight_max_abs_err={weight_max_abs_err:.6e} bias_abs_err={bias_abs_err:.6e}")


if __name__ == "__main__":
    main()
