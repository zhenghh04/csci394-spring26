# Tensor Parallel Exercise — Answers

Source: [`../../tensor_parallel/matmul_tensor_parallel.py`](../../tensor_parallel/matmul_tensor_parallel.py)
Extension: [`matmul_tensor_parallel_ext.py`](matmul_tensor_parallel_ext.py) — adds CLI args for `M, K, N` and a sweep mode.

The script computes `C = A @ B` where `A : [M, K]`, `B : [K, N]`. The shared
`K` dimension is sharded across `world_size` ranks; each rank multiplies its
`A[:, k_start:k_end] @ B[k_start:k_end, :]` and the partial `[M, N]` results
are summed via `all_reduce(SUM)`.

## Q1. Which dimension is sharded across ranks?

The contraction dimension `K`. In code (`matmul_tensor_parallel.py:95-100`):

```python
shard_size = K // world_size
k_start = rank * shard_size
A_shard = A_full[:, k_start:k_end]   # [M, K/P]
B_shard = B_full[k_start:k_end, :]   # [K/P, N]
```

Each rank holds a `K/P`-wide vertical strip of `A` and a matching
horizontal strip of `B`. `M` and `N` are not sharded — every rank produces
a full `[M, N]` partial result.

## Q2. Why does this implementation use `all_reduce(SUM)`?

A matrix product over a contracted index is a sum:

```
C[i, j] = Σ_k  A[i, k] * B[k, j]
        = Σ_p  ( Σ_{k ∈ shard_p}  A[i, k] * B[k, j] )
        = Σ_p  local_C_p[i, j]
```

Each rank computes one inner sum (its `local_C_p`), and the outer sum across
ranks is exactly an element-wise sum of `[M, N]` tensors. That is what
`all_reduce(SUM)` does — and afterward every rank holds the full result.

A `gather` would not work because the shapes are not concatenable: the
partial results are stacked in a depth direction, not appended along a row
or column. You could `reduce` to a single rank and then broadcast, but
`all_reduce` does both in one fused collective and is what NCCL is
optimized for.

## Q3. What would change if `B` were split by columns instead?

This is the column-parallel pattern: `B` is sharded along `N` instead of
`K`, and `A` is replicated.

```
A : [M, K]  (replicated)
B : [K, N]  →  B_p : [K, N/P]   (each rank owns a column block)

local_C_p = A @ B_p         # [M, N/P]
C = concat_along_N([local_C_p, ...])   # [M, N]
```

Three concrete differences:

1. **Replication pattern flips.** `A` is replicated; `B` is sharded.
2. **No reduction needed.** Each rank already produces final values; the
   contracted index `K` is computed in full locally. The partial output
   shapes are disjoint slices of the result.
3. **Collective changes from all-reduce to all-gather.** To materialize the
   full `[M, N]` on every rank you `all_gather` the per-rank `[M, N/P]`
   pieces along dimension 1.

Memory and compute are the same per rank (`M·K + K·(N/P)` weights,
`M·(N/P)·K` flops). The choice between K-shard (this exercise) and
N-shard depends on what the *next* layer expects: the column-parallel +
row-parallel pair used in transformers (see the 3D parallel exercise)
exists exactly so that the consumer of the matmul output is happy with
sharded inputs and one collective is enough per pair of layers.

## Q4. How do local matmul time and all-reduce time change as GPU count increases?

For `M = K = N = 16384` (the script default):

- **Local matmul time** scales as `1/P`: each rank does `M · (K/P) · N`
  flops, so doubling `P` halves the local compute time. On A100 with FP32
  this is ~600 ms at `P=1` and ~150 ms at `P=4` — assuming the matmul is
  not memory-bound, which at 16k it isn't.

- **All-reduce time** is *independent* of `P` for fixed message size.
  The partial result is `[M, N] = [16384, 16384]` FP32, i.e. 1 GiB per
  rank. NCCL's ring all-reduce moves `2 · (P-1)/P` of that volume per
  link, which for large `P` approaches `2 · M · N · 4 bytes` ≈ 2 GiB.
  On NVLink/NVSwitch this is dominated by per-stage bandwidth, not by
  the small log-N latency term — so total time stays roughly flat as `P`
  grows from 2 to 8.

Crossover behavior:

- At `P = 1`: local matmul = full cost, all-reduce = 0 → trivially fast.
- At `P = 2-4`: local matmul still dominates — TP wins.
- At `P >> 1` (e.g. crossing nodes via slower interconnect): all-reduce
  becomes the new floor and adding more ranks no longer reduces total
  time. This is the classic "communication wall" you can demonstrate
  with the extension by sweeping smaller `M, K, N`. With `M = K = N =
  2048` the local matmul is small enough that `t_allreduce > t_matmul`
  even at `P = 2` on a single node.

The extension script writes a CSV with `t_matmul_ms`, `t_allreduce_ms`,
and `wall_ms`, which makes this crossover easy to plot.

## Q5. Why does this demo still allocate the full matrices on every rank?

For correctness checking. The verification step on rank 0
(`matmul_tensor_parallel.py:193-203`) computes a reference `A_full @ B_full`
in one call and compares it element-wise to the gathered tensor-parallel
result. If `A` and `B` were already sharded, rank 0 would have to
all-gather them first — adding an `all_gather` to the timed region — or
write a separate single-process verifier.

In a real training system the shards are produced by sharded **weight
initialization** (each rank's TP linear layer instantiates only its
slice of the weight, never the full tensor). The matrices are never
materialized in full anywhere. The downside is that you can no longer
sanity-check against a reference matmul — you trust the autograd
contracts of `ColumnParallelLinear` / `RowParallelLinear` instead.

## Extension: parameterized matmul with a sweep

The extension in [`matmul_tensor_parallel_ext.py`](matmul_tensor_parallel_ext.py)
adds:

- `--M`, `--K`, `--N` CLI flags (with `--size` shortcut for the symmetric
  case);
- `--repeats` to time-average each measurement;
- a `--csv` argument that writes one row per run;
- the same correctness check (toggle with `--no-verify` for very large
  problems where the reference matmul does not fit on one device).

Recommended sweep on Polaris (4 GPUs, single node):

```bash
for SZ in 1024 2048 4096 8192 16384 24576; do
  torchrun --standalone --nproc_per_node=4 matmul_tensor_parallel_ext.py \
    --size "$SZ" --repeats 5 --csv results/sweep.csv
done
```

What to look for in `results/sweep.csv`:

- `t_matmul_ms` halves every time `P` doubles.
- `t_allreduce_ms` grows roughly linearly with `M · N` (message size) but
  is independent of `P`.
- The ratio `t_allreduce_ms / t_matmul_ms` crosses 1 around `M = K = N ≈
  4 k` on `P = 4` A100s — below that, communication dominates and TP is
  not worth doing on this size.
