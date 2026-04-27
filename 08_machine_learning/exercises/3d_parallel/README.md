# Exercise: 3D Parallel Transformer Training

This exercise benchmarks a small GPT-style transformer with three parallelism
dimensions:

- tensor parallelism (TP): split attention heads and feed-forward features
  within each transformer layer
- pipeline parallelism (PP): split transformer layers across stages
- data parallelism (DP): replicate the TP x PP model partition across groups

The script uses synthetic token data so the benchmark focuses on communication,
memory layout, and throughput rather than dataset loading.

## Files

| File | Description |
|---|---|
| `train_transformer_3d_parallel.py` | TP + PP + DP transformer throughput benchmark. |
| `launcher.sh` | Polaris/PBS helper that maps MPI rank variables to PyTorch distributed variables. |
| `run.sh` | Example 8-GPU sweep over several TP and PP configurations. |

## Rank Layout

The global rank is interpreted as:

```text
rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank
```

For example, with `tp_size = 2`, `pp_size = 2`, and `world_size = 8`,
the data-parallel size is:

```text
dp_size = world_size / (tp_size * pp_size) = 2
```

Each rank belongs to one TP group, one PP group, and one DP group.

## Run On Polaris interactively

```bash
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 \
  ./launcher.sh python3 train_transformer_3d_parallel.py \
  --tp-size 4 --pp-size 2
```

## Output

Rank 0 prints:

- TP, PP, DP, and world size
- rank-to-coordinate mapping
- average step time
- global tokens per second
- tokens per second per GPU

It also appends a row to:

```text
results_3d_parallel/scaling_results.csv
```

## Valid Configurations

The script checks several divisibility requirements:

- `tp_size * pp_size` must divide `world_size`.
- `num_heads` must be divisible by `tp_size`.
- `d_model` and `ff_dim` must be compatible with `tp_size`.
- `num_layers` must be divisible by `pp_size`.

## Questions

1. Which configurations improve global tokens per second?
2. Which configurations reduce tokens per second per GPU?
3. Why does pipeline parallelism introduce idle time in this script?
4. What communication operation is used inside tensor-parallel row-parallel
   layers?
5. How does DP differ from TP and PP in what it replicates or shards?

## Notes

- The pipeline schedule is intentionally simple and uses one micro-batch.
- Production pipeline training often uses 1F1B or interleaved schedules to
  reduce pipeline bubbles.
- The benchmark uses synthetic data and is meant for scaling analysis, not
  model-quality evaluation.
