# 3D Parallel Exercise — Answers

Source: [`../../3d_parallel/train_transformer_3d_parallel.py`](../../3d_parallel/train_transformer_3d_parallel.py)

The script benchmarks an 8-layer GPT-style transformer with three
parallelism dimensions composed on a 3D rank mesh:

```
rank = dp_rank · (pp_size · tp_size) + pp_rank · tp_size + tp_rank
dp_size = world_size / (tp_size · pp_size)
```

For 8 GPUs the compositions tested by the bundled `run.sh` are:

| TP | PP | DP | Comment |
|---:|---:|---:|---|
| 1 | 1 | 8 | pure DP — gradient all-reduce only |
| 2 | 1 | 4 | TP within node, DP across replicas |
| 2 | 4 | 1 | TP × deep PP — no DP |
| 1 | 8 | 1 | one layer per stage, no TP, no DP |

## Recommended sweep

`run_polaris.sh` runs all four configurations above plus three intermediates
(`tp=4 pp=1`, `tp=4 pp=2`, `tp=2 pp=2`). Each row is appended to
`results_3d_parallel/scaling_results.csv`.

## Q1. Which configurations improve global tokens per second?

For this small model (`d_model=512, num_heads=8, ff_dim=1024, 8 layers,
seq=2048`), pure DP (`tp=1, pp=1`) gives the highest global throughput.

Reasoning, in order of contribution:

- **DP is embarrassingly parallel.** Each replica processes its own
  micro-batch and the only communication is a single fused gradient
  all-reduce per step. With NCCL on NVLink the per-step overhead is small
  relative to compute, so 8× DP comes close to 8× the 1-GPU throughput.

- **TP scales poorly at small `d_model`.** The TP split only helps when
  the per-rank QKV/MLP matmuls are still large enough to keep an A100
  busy. With `d_model = 512`, `tp = 4` leaves each rank with `d_model/tp =
  128` per-head columns — small enough that kernel launch latency and the
  forward + backward all-reduce in `_AllReduceInForward` /
  `_AllReduceInBackward` (one each per layer) start to dominate. Net
  effect on this config: TP > 2 *reduces* global tokens/s.

- **PP serializes the forward and backward passes.** With one micro-batch
  per step (`pipeline_step` uses no micro-batching), `pp_size = K` adds a
  pipeline bubble of fraction `(K-1)/K` to every step. So `pp = 8` runs at
  ~12% efficiency per stage (87.5% bubble). The bigger PP, the worse the
  global throughput in this benchmark.

Configurations that *can* improve global tokens/s:

1. Increasing DP at fixed TP × PP (more replicas, more parallel batches).
2. Switching from PP to DP whenever the model fits in one GPU (PP is for
   when the model does not fit, not for speed).

## Q2. Which configurations reduce tokens per second per GPU?

Per-GPU throughput is the right metric for "is this configuration
efficient?". It drops whenever a non-DP dimension is added:

- `tp = 1, pp = 1, dp = 8`: highest per-GPU rate (set as 100% reference).
- `tp = 2, pp = 1, dp = 4`: ~70-85% of reference. Loss = forward +
  backward all-reduce in every TP-aware linear layer.
- `tp = 4, pp = 1, dp = 2`: ~55-70%. Communication grows with `tp_size`
  and per-rank matmuls shrink.
- `tp = 1, pp = 8, dp = 1`: ~12-15%. Pipeline bubble dominates because
  `pipeline_step` is sequential GPipe with one micro-batch.
- `tp = 2, pp = 4, dp = 1`: ~10-12%. Worst of both worlds at this scale.

The general pattern: TP and PP both *trade per-GPU efficiency for the
ability to hold a larger model*. They look like throughput losses on a
small model that already fits, and only become net wins at the scale where
DP alone runs out of memory.

## Q3. Why does pipeline parallelism introduce idle time in this script?

Look at `pipeline_step` (`train_transformer_3d_parallel.py:433-519`). The
forward and backward passes use **blocking** `dist.send` / `dist.recv` and
process **one micro-batch** per step. The execution order on `pp_size = K`
stages is:

```
time → →
stage 0:  [F0]                        [B0]
stage 1:        [F1]            [B1]
stage 2:              [F2]  [B2]
                       ...
stage K:                    [FK + loss + BK]
```

Stage 0 idles while stages 1…K finish their forwards, then idles again
while stages K…2 propagate gradients backward. The "bubble fraction" is
`(K-1) / K`, i.e. 50% bubble at `pp=2`, 75% at `pp=4`, 87.5% at `pp=8`.

To eliminate this you need either:

- **Micro-batching with 1F1B** — split the global batch into M
  micro-batches and interleave forward/backward across stages so each
  stage is busy for `M / (M + K - 1)` of the step. With `M ≫ K` the
  bubble shrinks to 0.
- **Interleaved 1F1B** (Megatron-LM style) — give each stage multiple
  non-contiguous layer chunks so it has work to do during what would have
  been bubble time.

The script comment block at lines 33-38 calls this out explicitly.

## Q4. What communication operation is used inside tensor-parallel row-parallel layers?

`all_reduce(SUM)` across the TP group, applied to the **output** of the
local matmul in the forward pass. See `RowParallelLinear.forward` at
`train_transformer_3d_parallel.py:261-266`:

```python
partial = self.linear(x)                                  # [B, T, out_features]
out = _AllReduceInForward.apply(partial, self.tp_group)   # all-reduce fwd, identity bwd
```

The complementary `ColumnParallelLinear` is the dual: identity in forward,
all-reduce in backward (so that the gradient reaching the previous layer
is the full gradient, not a per-rank shard). Together
`Column → Row` (used in attention QKV→out and MLP fc1→fc2) costs *one*
all-reduce per pair in each direction, regardless of how many TP ranks
there are. That is the key trick that makes Megatron-style TP affordable.

## Q5. How does DP differ from TP and PP in what it replicates or shards?

Concise version:

| Dim | What's sharded | What's replicated | Per-step communication |
|---|---|---|---|
| DP | the input batch | model weights | one all-reduce of full gradient set |
| TP | model weight matrices (along one axis) | input activation | 4 small all-reduces per layer (1 fwd + 1 bwd in each of 2 linears) |
| PP | model layers (along depth) | nothing — each stage owns a disjoint set | one send + recv per stage boundary, per direction |

Concrete consequences for this script:

- DP all-reduce happens once per step, after the full backward pass
  (`_dp_grad_sync`). It scales the gradient by `1/dp_size` so the
  optimizer sees the mean.
- TP all-reduces are inside `forward` / `backward` and happen `O(num_layers)`
  times per step, but each message is small (`B × T × d_model`).
- PP point-to-point messages move the inter-stage activation
  (`act_shape = [B, T, d_model]`) and its gradient — same shape, twice
  per stage boundary per step.

What this means for memory:
- DP: full model on every rank → memory does *not* scale.
- TP: each rank holds `1/tp_size` of every weight matrix → memory scales
  with TP.
- PP: each rank holds `1/pp_size` of the layers → memory scales with PP.

The composition `tp × pp` shrinks the per-rank **model** by `tp × pp`,
which is why both are needed before DP becomes the only valid axis for
scaling further.

## Notes on the script

- The pipeline bubble is intentionally not hidden so the cost is visible
  in the benchmark numbers (`train_transformer_3d_parallel.py:33-38`).
- `_dp_grad_sync` divides gradients by `dp_size`. If you compare losses
  across `dp` settings, use the per-step loss for a fixed batch size;
  `dp` simply averages over more samples per step.
- `act_shape` in the PP path is hard-coded from `(batch, seq, d_model)`.
  Mixed precision or sharded sequences would require a different
  exchange shape.
- `is_tp_sharded` is set on weights so a smarter DP-grad-sync routine
  could skip the all-reduce on TP-sharded weights (which are not
  replicated and therefore should *not* be averaged across DP). The
  current `_dp_grad_sync` all-reduces all params unconditionally — fine
  for this benchmark, but a real implementation would want to avoid the
  extra traffic on TP-sharded weights.
