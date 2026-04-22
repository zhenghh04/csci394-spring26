"""
Tiny decoder-only language model example for classroom DDP demos.

This script uses a built-in character-level corpus so students can run it
without downloading a dataset. It supports:

- single-process CPU or GPU runs
- multi-process DDP runs with torchrun
- a short text generation demo after training

Example commands:

    python3 train_tiny_llm_ddp.py --cpu --epochs 10
    torchrun --standalone --nproc_per_node=2 train_tiny_llm_ddp.py --cpu --epochs 10
    torchrun --standalone --nproc_per_node=2 train_tiny_llm_ddp.py --epochs 10
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


CORPUS = """
large language models are trained with next token prediction.
distributed training lets many accelerators work on the same optimization problem.
data parallelism replicates the model and splits the data across ranks.
tensor parallelism splits the math inside a layer across accelerators.
pipeline parallelism splits the model depth across stages.
fsdp and zero reduce memory by sharding model states across processes.
hpc systems matter because training needs compute memory bandwidth and communication.
students can study the same ideas with a much smaller transformer.
""".strip().lower() * 20


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[ch] for ch in text.lower() if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[idx] for idx in ids)


class NextTokenDataset(Dataset):
    def __init__(self, token_ids, seq_len, step=1, max_examples=None):
        self.examples = []
        limit = len(token_ids) - seq_len
        for start in range(0, max(limit, 0), step):
            chunk = token_ids[start:start + seq_len + 1]
            if len(chunk) < seq_len + 1:
                break
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.examples.append((x, y))
            if max_examples is not None and len(self.examples) >= max_examples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.transformer(h, mask=causal_mask)
        h = self.norm(h)
        return self.lm_head(h)


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(use_cpu):
    if not is_distributed():
        if torch.cuda.is_available() and not use_cpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return 0, 1, 0, device

    backend = "nccl" if torch.cuda.is_available() and not use_cpu else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available() and not use_cpu:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def build_dataloader(dataset, batch_size, rank, world_size, use_distributed):
    sampler = None
    shuffle = True
    if use_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    return loader, sampler


def train_one_epoch(model, loader, optimizer, device, vocab_size, rank, world_size):
    model.train()
    running_loss = 0.0
    running_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()

        local_tokens = y.numel()
        running_loss += loss.item() * local_tokens
        running_tokens += local_tokens

    stats = torch.tensor([running_loss, running_tokens], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    avg_loss = stats[0].item() / stats[1].item()
    if rank == 0:
        print(f"  average token loss = {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def generate_text(model, tokenizer, device, prompt, max_new_tokens):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        window = x[:, -model.seq_len:]
        logits = model(window)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, next_token], dim=1)
    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--step", type=int, default=8, help="stride between training windows")
    parser.add_argument("--max-examples", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="force CPU and gloo backend")
    parser.add_argument("--prompt", type=str, default="distributed training ")
    parser.add_argument("--generate-tokens", type=int, default=80)
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed(args.cpu)
    torch.manual_seed(args.seed + rank)

    tokenizer = CharTokenizer(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    dataset = NextTokenDataset(
        token_ids=token_ids,
        seq_len=args.seq_len,
        step=args.step,
        max_examples=args.max_examples,
    )
    loader, sampler = build_dataloader(dataset, args.batch_size, rank, world_size, is_distributed())

    model = TinyCausalLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    if is_distributed():
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        print("=== Tiny LLM DDP Demo ===")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Training sequences: {len(dataset)}")
        print(f"Batch size per rank: {args.batch_size}")
        print(f"Global batch size: {args.batch_size * world_size}")
        print()

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Epoch {epoch}")
        train_one_epoch(model, loader, optimizer, device, tokenizer.vocab_size, rank, world_size)

    if rank == 0:
        elapsed = time.time() - start
        model_for_generation = model.module if isinstance(model, DDP) else model
        sample = generate_text(
            model_for_generation,
            tokenizer,
            device,
            prompt=args.prompt,
            max_new_tokens=args.generate_tokens,
        )
        print()
        print(f"Training time: {elapsed:.2f}s")
        print("Generated sample:")
        print(sample)

    cleanup_distributed()


if __name__ == "__main__":
    main()
