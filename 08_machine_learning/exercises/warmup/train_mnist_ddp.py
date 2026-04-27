"""
MNIST DDP Training Script
Distributed Data Parallel training of a CNN on MNIST.
Tracks throughput (samples/sec) and timing for scaling analysis.

Lines marked with #DDP are the changes needed to convert single-GPU to multi-GPU.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist                              #DDP: distributed communication
from torch.nn.parallel import DistributedDataParallel as DDP  #DDP: model wrapper
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler   #DDP: data partitioning
from torchvision import datasets, transforms
import time
import os
import csv
import argparse


class MNISTNet(nn.Module):
    """Same model architecture as single-GPU — no changes needed."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, rank):
    """Training loop — identical to single-GPU. DDP handles gradient sync automatically."""
    model.train()
    total_loss = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()   # DDP hooks fire all-reduce here automatically
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    if rank == 0:                                              #DDP: only print on rank 0
        print(f"  Epoch {epoch}: avg train loss = {avg_loss:.4f}")
    return avg_loss


def test(model, device, test_loader, rank):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if rank == 0:                                              #DDP: only print on rank 0
        print(f"  Test: loss={test_loss:.4f}, accuracy={accuracy:.1f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup-epochs', type=int, default=0)  #DDP: warmup epochs (0=no warmup)
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    # ---- DDP: Initialize distributed process group ----
    # Creates a communication group so all GPUs can synchronize gradients.
    # The "nccl" backend is optimized for NVIDIA GPU-to-GPU communication.
    dist.init_process_group(backend="nccl")                    #DDP
    rank = dist.get_rank()                                     #DDP: global process ID
    world_size = dist.get_world_size()                         #DDP: total number of GPUs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))          #DDP: GPU index on this node
    device = torch.device(f"cuda:{local_rank}")                #DDP: each rank uses its own GPU
    torch.cuda.set_device(device)                              #DDP

    if rank == 0:
        print(f"=== MNIST DDP Training ===")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    train_dataset = datasets.MNIST(data_dir, train=True, download=(rank == 0), transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=(rank == 0), transform=transform)    
    dist.barrier()                                             #DDP: wait for rank 0 to download
    if rank != 0:
        train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)    

    # ---- DDP: Use DistributedSampler to partition data across GPUs ----
    # Each GPU gets a non-overlapping 1/N slice of the dataset.
    # shuffle=False because the sampler handles shuffling internally.
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=world_size,
                                       rank=rank)              #DDP
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False,                   #DDP: sampler handles shuffling
                              sampler=train_sampler,           #DDP
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ---- DDP: Wrap model with DistributedDataParallel ----
    # DDP replicates the model on each GPU and installs backward hooks
    # that automatically all-reduce gradients after each backward pass.
    model = MNISTNet().to(device)
    model = DDP(model, device_ids=[local_rank])                #DDP

    # ---- DDP: Scale learning rate with number of GPUs ----
    # With N GPUs, effective batch size is N× larger. To maintain the same
    # training dynamics, scale lr proportionally (linear scaling rule).
    scaled_lr = args.lr * world_size                           #DDP
    # If warmup_epochs > 0, start at base_lr and ramp to scaled_lr.    #DDP
    # During warmup, LR does NOT scale — it gradually increases.       #DDP
    initial_lr = args.lr if args.warmup_epochs > 0 else scaled_lr      #DDP
    if rank == 0:
        print(f"Base LR: {args.lr}, Scaled LR: {scaled_lr} (x{world_size})")
        if args.warmup_epochs > 0:
            print(f"Warmup: {args.warmup_epochs} epochs (lr: {args.lr} -> {scaled_lr})")
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Training loop with timing
    epoch_times = []
    total_samples = len(train_dataset)

    # Untimed warmup iteration (GPU/NCCL init)
    if rank == 0:
        print("Warmup iteration...")
    train_sampler.set_epoch(-1)                                #DDP: set epoch for shuffling
    train(model, device, train_loader, optimizer, 0, rank)
    torch.cuda.synchronize()
    dist.barrier()                                             #DDP: sync all GPUs

    if rank == 0:
        print(f"\nStarting timed training ({args.epochs} epochs)...")

    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # ---- DDP: LR warmup schedule ----                            #DDP
        # During warmup epochs, ramp LR from base_lr to scaled_lr.     #DDP
        # After warmup, use full scaled_lr.                             #DDP
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:     #DDP
            current_lr = args.lr + (scaled_lr - args.lr) * (epoch / args.warmup_epochs)
        else:                                                          #DDP
            current_lr = scaled_lr                                     #DDP
        for param_group in optimizer.param_groups:                      #DDP
            param_group['lr'] = current_lr                             #DDP

        train_sampler.set_epoch(epoch)                         #DDP: new shuffle each epoch
        torch.cuda.synchronize()
        epoch_start = time.time()

        train(model, device, train_loader, optimizer, epoch, rank)

        torch.cuda.synchronize()
        dist.barrier()                                         #DDP: wait for all GPUs
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        accuracy = test(model, device, test_loader, rank)

        if rank == 0:                                          #DDP: only rank 0 reports
            throughput = total_samples / epoch_time
            print(f"  lr={current_lr:.6f} | Epoch time: {epoch_time:.2f}s | Throughput: {throughput:.0f} samples/sec\n")

    total_time = time.time() - total_start

    # Report results (only rank 0 writes)
    if rank == 0:                                              #DDP
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        avg_throughput = total_samples / avg_epoch_time
        print(f"{'='*50}")
        print(f"MNIST DDP Results ({world_size} GPUs)")
        print(f"  Avg epoch time:  {avg_epoch_time:.2f}s")
        print(f"  Avg throughput:  {avg_throughput:.0f} samples/sec")
        print(f"  Total time:      {total_time:.2f}s")
        print(f"  Final accuracy:  {accuracy:.1f}%")
        print(f"{'='*50}")

        # Save results to CSV
        os.makedirs(args.results_dir, exist_ok=True)
        csv_path = os.path.join(args.results_dir, "scaling_results.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['model', 'world_size', 'batch_size_per_gpu',
                                 'effective_batch_size', 'base_lr', 'scaled_lr',
                                 'avg_epoch_time_s',
                                 'avg_throughput_samples_per_sec', 'total_time_s',
                                 'final_accuracy_pct'])
            writer.writerow(['mnist', world_size, args.batch_size,
                             args.batch_size * world_size,
                             f"{args.lr}", f"{scaled_lr}",
                             f"{avg_epoch_time:.3f}",
                             f"{avg_throughput:.1f}", f"{total_time:.3f}",
                             f"{accuracy:.1f}"])

    dist.destroy_process_group()                               #DDP: cleanup


if __name__ == "__main__":
    main()
