# MPI ping-pong

Two-rank latency/bandwidth test using blocking send/recv. Rank 0 sends a
message to rank 1 and receives it back; results are reported per message size.
This benchmark should be run on two different nodes (one rank per node).

## Bandwidth-only plot
Generate a bandwidth-only plot from benchmark output:

```bash
mpirun -np 2 --ppn 1 ./pingpong --min 1 --max 8M --iters 1000 --warmup 100 > pingpong.out
python3 plot_bandwidth.py --input pingpong.out --output pingpong_bandwidth.png --metric sum
```

This produces `pingpong_bandwidth.png` with only bandwidth vs message size.

## Build
```bash
make
```

## Run (example)
Request two nodes and place one rank per node.
```bash
mpirun -np 2 --ppn 1 ./pingpong --min 1 --max 8M --iters 1000 --warmup 100
```

## Options
- `--min SIZE` minimum message size (bytes or K/M/G suffix)
- `--max SIZE` maximum message size (bytes or K/M/G suffix)
- `--iters N` timed iterations (default 1000)
- `--warmup N` warmup iterations (default 100)

## Plot script options
- `--input FILE` benchmark text output file
- `--output FILE` output image path (default `pingpong_bandwidth.png`)
- `--metric sum|avg|min|max` choose bandwidth column (default `sum`)
