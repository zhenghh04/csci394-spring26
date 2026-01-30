# MPI ping-pong

Two-rank latency/bandwidth test using blocking send/recv. Rank 0 sends a
message to rank 1 and receives it back; results are reported per message size.

## Build
```bash
make
```

## Run (example)
```bash
mpirun -np 2 ./pingpong --min 1 --max 8M --iters 1000 --warmup 100
```

## Options
- `--min SIZE` minimum message size (bytes or K/M/G suffix)
- `--max SIZE` maximum message size (bytes or K/M/G suffix)
- `--iters N` timed iterations (default 1000)
- `--warmup N` warmup iterations (default 100)
