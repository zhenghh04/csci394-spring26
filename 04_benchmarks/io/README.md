# File-per-rank I/O test

Writes a fixed amount of data per rank to a separate file and reports aggregate
bandwidth using the slowest rank's time.

## Build
```bash
make
```

## Run (example)
```bash
mpirun -np 8 ./io_test --mb 256 --iters 1 --chunk-kb 1024 --dir ./io_out
```

## Options
- `--mb N` total MB written per rank per iteration (default 256)
- `--iters N` iterations (default 1)
- `--dir PATH` output directory (default ./io_out)
- `--chunk-kb N` chunk size in KB (default 1024)
- `--fsync` call fsync() before close
- `--keep` keep files after test (default: delete)
