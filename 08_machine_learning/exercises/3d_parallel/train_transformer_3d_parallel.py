"""
3D Parallelism Scaling Study: 8-Layer Transformer with TP + PP + DP
(CSCI 394 Spring 2026 — Distributed Deep Learning Tutorial)

Demonstrates how to combine three parallelism strategies:
  - Tensor Parallelism (TP): splits attention heads and FFN features across tp_size GPUs
  - Pipeline Parallelism (PP): assigns groups of layers to pp_size pipeline stages
  - Data Parallelism  (DP): replicates the TP×PP mesh across dp_size replica groups

Global rank layout (rank = dp * pp_size * tp_size + pp * tp_size + tp):

  Example: tp=2, pp=2, dp=2 on 8 GPUs
  rank  dp  pp  tp
    0    0   0   0   ← stage 0, DP replica 0
    1    0   0   1
    2    0   1   0   ← stage 1, DP replica 0
    3    0   1   1
    4    1   0   0   ← stage 0, DP replica 1
    5    1   0   1
    6    1   1   0   ← stage 1, DP replica 1
    7    1   1   1

  TP groups: [0,1], [2,3], [4,5], [6,7]   ← use fast NVLink within node
  PP groups: [0,2], [1,3], [4,6], [5,7]   ← send activations between stages
  DP groups: [0,4], [1,5], [2,6], [3,7]   ← all-reduce gradients

Communication primitives:
  TP: all-reduce at row-parallel output (forward); identity (backward)
      identity at column-parallel input (forward); all-reduce (backward)
  PP: blocking send/recv between neighboring pipeline stages
  DP: all-reduce of all gradients after backward pass

Note on pipeline schedule:
  This script uses a simple sequential GPipe-style schedule (one micro-batch,
  no stage overlap). The bubble fraction is (pp-1)/pp for pp stages.
  Production systems use 1F1B (interleaved) scheduling to reduce the bubble.
  For a throughput benchmark the relative numbers across configurations are
  still meaningful, and the sequential schedule is easier to read.

Usage (2 nodes × 4 GPUs = 8 GPUs on Polaris):

  mpiexec -n 8 --ppn 4 --hostfile $PBS_NODEFILE \\
    --env MASTER_ADDR=<node0> --env MASTER_PORT=29500 \\
    bash -lc "
      export RANK=$PMI_RANK LOCAL_RANK=$PMI_LOCAL_RANK WORLD_SIZE=8
      python train_transformer_3d_parallel.py --tp-size 2 --pp-size 2
    "

See run_3d_parallel_scaling.sh for a PBS script that sweeps all TP×PP combos.
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TP communication primitives (custom autograd functions)
# =============================================================================

class _AllReduceInBackward(torch.autograd.Function):
    """
    Forward : identity  (no communication).
    Backward: all-reduce across TP group (sums partial gradients).

    Apply to the INPUT of a column-parallel linear layer so that the gradient
    reaching the previous layer (e.g. LayerNorm) is the full gradient, not a
    per-rank partial.  This is the "conjugate" of the all-reduce in the row-
    parallel output.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


class _AllReduceInForward(torch.autograd.Function):
    """
    Forward : all-reduce across TP group (sums partial products).
    Backward: identity  (each rank receives the full gradient).

    Apply to the OUTPUT of a row-parallel linear layer to collapse the partial
    results from each TP rank into the full output.
    """
    @staticmethod
    def forward(ctx, x, group):
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None


# =============================================================================
# Process group management
# =============================================================================

class ParallelContext:
    """
    Creates and stores TP, PP, and DP process groups.

    Global rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank
    """

    def __init__(self, tp_size: int, pp_size: int):
        self.local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        dist.init_process_group(backend="nccl", device_id=self.device)
        self.global_rank = dist.get_rank()
        self.world_size  = dist.get_world_size()

        dp_size = self.world_size // (tp_size * pp_size)
        assert tp_size * pp_size * dp_size == self.world_size, (
            f"tp={tp_size} × pp={pp_size} must divide world_size={self.world_size}"
        )
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size

        # Decode this rank's (dp, pp, tp) coordinates
        r            = self.global_rank
        self.tp_rank = r % tp_size;   r //= tp_size
        self.pp_rank = r % pp_size;   r //= pp_size
        self.dp_rank = r

        def gr(dp, pp, tp):
            return dp * pp_size * tp_size + pp * tp_size + tp

        # --- TP groups: same (dp, pp), all tp values ---
        self.tp_group = None
        for dp in range(dp_size):
            for pp in range(pp_size):
                members = [gr(dp, pp, tp) for tp in range(tp_size)]
                g = dist.new_group(members)
                if self.global_rank in members:
                    self.tp_group = g

        # --- PP groups: same (dp, tp), all pp values ---
        # Also build a table: pp_stage → global rank (for this (dp, tp) slice)
        self.pp_group = None
        self._pp_stage_to_rank = {}
        for dp in range(dp_size):
            for tp in range(tp_size):
                members = [gr(dp, pp, tp) for pp in range(pp_size)]
                g = dist.new_group(members)
                if self.global_rank in members:
                    self.pp_group = g
                    self._pp_stage_to_rank = {pp: rank for pp, rank in enumerate(members)}

        # --- DP groups: same (pp, tp), all dp values ---
        self.dp_group = None
        for pp in range(pp_size):
            for tp in range(tp_size):
                members = [gr(dp, pp, tp) for dp in range(dp_size)]
                g = dist.new_group(members)
                if self.global_rank in members:
                    self.dp_group = g


    # Pipeline neighbor helpers
    def next_pp_rank(self):
        if self.pp_rank == self.pp_size - 1:
            return None
        return self._pp_stage_to_rank[self.pp_rank + 1]

    def prev_pp_rank(self):
        if self.pp_rank == 0:
            return None
        return self._pp_stage_to_rank[self.pp_rank - 1]

    def is_first_stage(self): return self.pp_rank == 0
    def is_last_stage(self):  return self.pp_rank == self.pp_size - 1

    def log(self, msg):
        if self.global_rank == 0:
            print(msg, flush=True)

    def destroy(self):
        dist.destroy_process_group()


# =============================================================================
# Tensor-parallel linear layers
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer whose OUTPUT features are split across tp_size ranks.

    Each rank holds weight of shape [out_total // tp_size, in_features].
    Input tensor x must be IDENTICAL on all TP ranks (replicated).

    Forward:  apply _AllReduceInBackward to x (no-op), then local matmul.
    Backward: autograd handles the local matmul; _AllReduceInBackward's
              backward all-reduces dL/dx across TP so that the full gradient
              reaches layers upstream of this one.
    """

    def __init__(self, in_features: int, out_features_total: int,
                 tp_group, bias: bool = True):
        super().__init__()
        tp_size = dist.get_world_size(tp_group)
        assert out_features_total % tp_size == 0, (
            f"out_features {out_features_total} must be divisible by tp_size {tp_size}"
        )
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features, out_features_total // tp_size, bias=bias)
        # Mark TP-sharded so the DP gradient sync knows not to skip these
        self.linear.weight.is_tp_sharded = True
        if bias:
            self.linear.bias.is_tp_sharded = True

    def forward(self, x):
        x = _AllReduceInBackward.apply(x, self.tp_group)  # identity fwd, all-reduce bwd
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """
    Linear layer whose INPUT features are split across tp_size ranks.

    Each rank holds weight of shape [out_features, in_total // tp_size].
    The input x on each rank is a DIFFERENT shard (complementary to
    column-parallel output from the preceding layer).

    Forward:  local matmul → all-reduce across TP to sum partial products.
    Backward: _AllReduceInForward's backward is identity, so dL/dy flows
              unchanged to each rank's local backward.
    """

    def __init__(self, out_features: int, in_features_total: int,
                 tp_group, bias: bool = True):
        super().__init__()
        tp_size = dist.get_world_size(tp_group)
        assert in_features_total % tp_size == 0, (
            f"in_features {in_features_total} must be divisible by tp_size {tp_size}"
        )
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features_total // tp_size, out_features, bias=False)
        self.linear.weight.is_tp_sharded = True
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        partial = self.linear(x)                                    # [B, T, out_features]
        out = _AllReduceInForward.apply(partial, self.tp_group)     # all-reduce fwd, id bwd
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# Transformer building blocks (TP-aware)
# =============================================================================

class TPSelfAttention(nn.Module):
    """
    Multi-head self-attention with tensor parallelism.

    Each TP rank handles (num_heads // tp_size) heads:
      QKV projection : ColumnParallelLinear  [d_model → 3 * (d_model / tp)]
      Attention       : local, no communication
      Output proj     : RowParallelLinear    [d_model / tp → d_model] + all-reduce
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float, tp_group):
        super().__init__()
        tp_size     = dist.get_world_size(tp_group)
        assert num_heads % tp_size == 0, (
            f"num_heads {num_heads} must be divisible by tp_size {tp_size}"
        )
        self.d_model         = d_model
        self.num_heads       = num_heads
        self.num_local_heads = num_heads // tp_size
        self.head_dim        = d_model // num_heads
        self.local_dim       = self.num_local_heads * self.head_dim
        self.scale           = self.head_dim ** -0.5
        self.dropout_p       = dropout

        # Column parallel: each rank produces 3 * local_dim outputs (Q, K, V)
        self.qkv_proj = ColumnParallelLinear(d_model, 3 * d_model, tp_group, bias=False)
        # Row parallel: reduce local_dim → d_model
        self.out_proj = RowParallelLinear(d_model, d_model, tp_group, bias=True)

    def forward(self, x, causal_mask):
        B, T, _ = x.shape

        # QKV: [B, T, 3 * local_dim]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.local_dim, dim=-1)

        def reshape(t):
            return t.view(B, T, self.num_local_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)  # [B, local_heads, T, head_dim]

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, local_heads, T, T]
        attn = attn + causal_mask                            # add -inf for future tokens
        attn = attn.softmax(dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = attn @ v                                       # [B, local_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.local_dim)  # [B, T, local_dim]

        # Row parallel projection + all-reduce
        return self.out_proj(out)                            # [B, T, d_model]


class TPMLP(nn.Module):
    """
    Position-wise feed-forward network with tensor parallelism.

      FC1: ColumnParallelLinear  [d_model → ff_dim / tp] + GELU
      FC2: RowParallelLinear     [ff_dim / tp → d_model] + all-reduce
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float, tp_group):
        super().__init__()
        self.fc1     = ColumnParallelLinear(d_model, ff_dim, tp_group, bias=True)
        # RowParallelLinear(out_features, in_features_total, ...):
        # FC2 maps ff_dim → d_model; each rank holds ff_dim//tp input features.
        self.fc2     = RowParallelLinear(d_model, ff_dim, tp_group, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: LN → Attention → residual → LN → MLP → residual."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float, tp_group):
        super().__init__()
        self.ln1   = nn.LayerNorm(d_model)
        self.attn  = TPSelfAttention(d_model, num_heads, dropout, tp_group)
        self.ln2   = nn.LayerNorm(d_model)
        self.mlp   = TPMLP(d_model, ff_dim, dropout, tp_group)

    def forward(self, x, causal_mask):
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# Pipeline stage: owns a subset of the 8 transformer layers
# =============================================================================

class PipelineStage(nn.Module):
    """
    One stage in the pipeline.

    Stage 0             : token embedding + position embedding + transformer blocks
    Intermediate stages : transformer blocks only
    Last stage          : transformer blocks + final LayerNorm + LM head (vocab projection)
    """

    def __init__(self, ctx: ParallelContext,
                 vocab_size: int, d_model: int, num_heads: int,
                 ff_dim: int, num_layers: int, seq_len: int, dropout: float):
        super().__init__()
        assert num_layers % ctx.pp_size == 0, (
            f"num_layers {num_layers} must be divisible by pp_size {ctx.pp_size}"
        )
        layers_per_stage = num_layers // ctx.pp_size

        # Build this stage's transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout, ctx.tp_group)
            for _ in range(layers_per_stage)
        ])

        if ctx.is_first_stage():
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

        if ctx.is_last_stage():
            self.final_ln = nn.LayerNorm(d_model)
            self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        self.d_model  = d_model
        self.seq_len  = seq_len
        self.device   = ctx.device

    def forward(self, x):
        """
        x : [B, T] int64 tokens if first stage, else [B, T, d_model] float.
        Returns [B, T, d_model] for non-last stages, [B, T, vocab_size] for last.
        """
        if hasattr(self, 'tok_emb'):
            B, T = x.shape
            pos  = torch.arange(T, device=x.device)
            x    = self.tok_emb(x) + self.pos_emb(pos)         # [B, T, d_model]

        B, T, _ = x.shape
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1,
        )  # [T, T]

        for block in self.blocks:
            x = block(x, causal_mask)

        if hasattr(self, 'lm_head'):
            x = self.lm_head(self.final_ln(x))                  # [B, T, vocab_size]
        return x


# =============================================================================
# One training step with pipeline parallelism
# =============================================================================

def pipeline_step(ctx, stage, optimizer,
                  tokens, targets,
                  act_shape, vocab_size):
    """
    Forward pass  → stage 0 → stage 1 → ... → stage pp-1 (loss)
    Backward pass → stage pp-1 → ... → stage 1 → stage 0

    Blocking send/recv creates the natural ordering:
      stage 0 : forward → send → [blocks waiting for grad] → backward
      stage 1 : recv → forward → send → recv grad → backward → send grad
      ...
      stage K : recv → forward → loss.backward() → send grad

    After backward, gradients are all-reduced across the DP group.
    Returns loss_value (float) for the last stage, None for all other stages.
    Timing is intentionally left to the caller so that torch.cuda.synchronize()
    brackets the measurement correctly.
    """
    d = ctx.device

    # ------------------------------------------------------------------
    # pp_size == 1: no pipeline communication, just a standard step
    # ------------------------------------------------------------------
    if ctx.pp_size == 1:
        logits = stage(tokens)                                   # [B, T, V]
        loss   = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        _dp_grad_sync(ctx, stage)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    # ------------------------------------------------------------------
    # pp_size > 1: pipeline communication via blocking send/recv
    # ------------------------------------------------------------------

    # ---------- FORWARD ----------
    if ctx.is_first_stage():
        out      = stage(tokens)                                 # [B, T, d_model]
        dist.send(out.detach().contiguous(), dst=ctx.next_pp_rank())
        saved_out  = out          # keep the graph alive for backward
        saved_recv = None

    elif ctx.is_last_stage():
        buf = torch.empty(act_shape, dtype=torch.float32, device=d)
        dist.recv(buf, src=ctx.prev_pp_rank())
        buf.requires_grad_(True)
        logits = stage(buf)                                      # [B, T, V]
        loss   = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        saved_recv = buf
        saved_out  = None

    else:  # intermediate stage
        buf = torch.empty(act_shape, dtype=torch.float32, device=d)
        dist.recv(buf, src=ctx.prev_pp_rank())
        buf.requires_grad_(True)
        out = stage(buf)
        dist.send(out.detach().contiguous(), dst=ctx.next_pp_rank())
        saved_recv = buf
        saved_out  = out

    # ---------- BACKWARD ----------
    if ctx.is_last_stage():
        loss.backward()
        # The gradient wrt the activation received from the previous stage
        dist.send(saved_recv.grad.contiguous(), dst=ctx.prev_pp_rank())
        loss_val = loss.item()

    elif ctx.is_first_stage():
        grad_buf = torch.empty(act_shape, dtype=torch.float32, device=d)
        dist.recv(grad_buf, src=ctx.next_pp_rank())
        saved_out.backward(gradient=grad_buf)
        loss_val = None

    else:  # intermediate
        grad_buf = torch.empty_like(saved_out)
        dist.recv(grad_buf, src=ctx.next_pp_rank())
        saved_out.backward(gradient=grad_buf)
        dist.send(saved_recv.grad.contiguous(), dst=ctx.prev_pp_rank())
        loss_val = None

    # ---------- DP GRADIENT ALL-REDUCE ----------
    _dp_grad_sync(ctx, stage)
    optimizer.step()
    optimizer.zero_grad()

    return loss_val


def _dp_grad_sync(ctx, stage):
    """All-reduce gradients across the DP group, then scale by 1/dp_size."""
    if ctx.dp_size == 1:
        return
    for param in stage.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=ctx.dp_group)
            param.grad.div_(ctx.dp_size)


# =============================================================================
# Main: benchmark TP + PP + DP throughput
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D Parallelism Throughput Benchmark")

    # Parallelism config
    parser.add_argument("--tp-size",   type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--pp-size",   type=int, default=1, help="Pipeline parallelism degree")

    # Model config
    parser.add_argument("--vocab-size",  type=int, default=32000)
    parser.add_argument("--d-model",     type=int, default=2048)
    parser.add_argument("--num-heads",   type=int, default=8)
    parser.add_argument("--ff-dim",      type=int, default=2048)
    parser.add_argument("--num-layers",  type=int, default=8)
    parser.add_argument("--seq-len",     type=int, default=8192)
    parser.add_argument("--dropout",     type=float, default=0.0,
                        help="Set to 0 for a deterministic throughput benchmark")

    # Training / benchmark config
    parser.add_argument("--batch-size",  type=int, default=4,
                        help="Token sequences per DP rank per step")
    parser.add_argument("--num-iters",   type=int, default=20,
                        help="Total iterations (first 5 are warmup)")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--lr",          type=float, default=1e-4)

    # Output
    parser.add_argument("--results-dir", type=str, default="./results_3d_parallel")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    ctx = ParallelContext(tp_size=args.tp_size, pp_size=args.pp_size)

    cfg_str = (f"TP={args.tp_size} PP={args.pp_size} DP={ctx.dp_size} "
               f"(world={ctx.world_size})")
    ctx.log(f"\n{'='*60}")
    ctx.log(f"  3D Parallelism Benchmark — {cfg_str}")
    ctx.log(f"  Model  : {args.num_layers}L × d={args.d_model} "
            f"× {args.num_heads}H × ff={args.ff_dim} × seq={args.seq_len}")
    ctx.log(f"  Batch  : {args.batch_size} seqs/rank "
            f"→ {args.batch_size * ctx.dp_size} seqs/step (global)")
    ctx.log(f"{'='*60}")

    # Print one line per rank to verify the coordinate assignment
    for r in range(ctx.world_size):
        if ctx.global_rank == r:
            print(f"  rank {r:2d}  dp={ctx.dp_rank} pp={ctx.pp_rank} tp={ctx.tp_rank}"
                  f"  local_gpu={ctx.local_rank}", flush=True)
        dist.barrier()

    # ------------------------------------------------------------------
    # Build model stage for this rank
    # ------------------------------------------------------------------
    stage = PipelineStage(
        ctx        = ctx,
        vocab_size = args.vocab_size,
        d_model    = args.d_model,
        num_heads  = args.num_heads,
        ff_dim     = args.ff_dim,
        num_layers = args.num_layers,
        seq_len    = args.seq_len,
        dropout    = args.dropout,
    ).to(ctx.device)

    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr)

    # Shape of the inter-stage activation tensor
    act_shape = (args.batch_size, args.seq_len, args.d_model)

    # ------------------------------------------------------------------
    # Synthetic data (each DP rank uses the same random seed here; in real
    # training each DP rank would receive a different mini-batch)
    # ------------------------------------------------------------------
    torch.manual_seed(42 + ctx.dp_rank)

    def make_batch():
        tokens  = torch.randint(0, args.vocab_size,
                                (args.batch_size, args.seq_len),
                                device=ctx.device)
        targets = torch.randint(0, args.vocab_size,
                                (args.batch_size, args.seq_len),
                                device=ctx.device)
        return tokens, targets

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    ctx.log(f"\nWarmup ({args.warmup_iters} iters) ...")
    for _ in range(args.warmup_iters):
        tokens, targets = make_batch()
        pipeline_step(ctx, stage, optimizer, tokens, targets,
                      act_shape, args.vocab_size)
    torch.cuda.synchronize()
    dist.barrier()

    # ------------------------------------------------------------------
    # Benchmark — synchronize GPU before and after each step so that the
    # elapsed time includes all kernel execution, not just kernel launch.
    # ------------------------------------------------------------------
    ctx.log(f"Benchmarking ({args.num_iters - args.warmup_iters} iters) ...")
    iter_times = []
    for i in range(args.num_iters - args.warmup_iters):
        tokens, targets = make_batch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_val = pipeline_step(ctx, stage, optimizer, tokens, targets,
                                 act_shape, args.vocab_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        iter_times.append(elapsed)
        if ctx.global_rank == 0 and i % 5 == 0:
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A (not last stage)"
            print(f"  iter {i:3d}  loss={loss_str}  step_time={elapsed*1000:.1f}ms",
                  flush=True)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    avg_step_s   = sum(iter_times) / len(iter_times)
    # Global tokens per step = batch_size * seq_len * dp_size
    global_tokens_per_step = args.batch_size * args.seq_len * ctx.dp_size
    tokens_per_s           = global_tokens_per_step / avg_step_s

    ctx.log(f"\n{'='*60}")
    ctx.log(f"  Results — {cfg_str}")
    ctx.log(f"  Avg step time     : {avg_step_s*1000:.2f} ms")
    ctx.log(f"  Global tokens/sec : {tokens_per_s:,.0f}")
    ctx.log(f"  Tokens/sec/GPU    : {tokens_per_s / ctx.world_size:,.0f}")
    ctx.log(f"{'='*60}\n")

    # Write CSV summary (rank 0 only)
    if ctx.global_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        summary_path = os.path.join(args.results_dir, "scaling_results.csv")
        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "tp_size", "pp_size", "dp_size", "world_size",
                    "d_model", "num_heads", "ff_dim", "num_layers", "seq_len",
                    "batch_per_dp", "global_tokens_per_step",
                    "avg_step_ms", "tokens_per_s", "tokens_per_s_per_gpu",
                ])
            w.writerow([
                args.tp_size, args.pp_size, ctx.dp_size, ctx.world_size,
                args.d_model, args.num_heads, args.ff_dim, args.num_layers, args.seq_len,
                args.batch_size, global_tokens_per_step,
                f"{avg_step_s*1000:.2f}", f"{tokens_per_s:.0f}",
                f"{tokens_per_s / ctx.world_size:.0f}",
            ])
        print(f"Results appended to {summary_path}", flush=True)

    ctx.destroy()


if __name__ == "__main__":
    main()
