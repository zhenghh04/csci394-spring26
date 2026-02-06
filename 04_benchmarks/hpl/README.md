# HPL (High Performance Linpack)

HPL is typically provided by the supercomputer as an optimized binary. This
folder includes a template `HPL.dat` and a sample PBS script for running HPL.

## Build (from source with provided script)
This folder includes a helper script that downloads HPL from Netlib and builds
Building hpl on crux
```bash
sh build_crux.sh
```

## Running HPL
### Notes
- The `P x Q` process grid in `HPL.dat` must multiply to the number of ranks.
- Problem size `N` should be large enough to fill most of the node memory.

### `hpl_size.py` helper
`hpl_size.py` computes one or more suggested HPL problem sizes (`N`) from
available memory and node count. It assumes HPL uses ~8 bytes per matrix entry
and rounds `N` down to a multiple of `NB` (default `384`).

Usage highlights:
- `--mem`: memory per node by default (e.g., `256GB`, `256GiB`). Use `--total-mem`
  if you want to pass total cluster memory instead.
- `--utilization`: fraction of total memory to target (default `0.60`).
- `--count` and `--step-percent`: emit multiple sizes by decreasing utilization
  each step (e.g., utilization, then utilization - step%, etc.).

### Steps
1. Edit `HPL.dat` to match the number of MPI ranks and problem sizes you want.
    - Please run ``python hpl_size.py --mem 256gb --num-nodes NUM_NODES --count 3`` to get suggest problem sizes. 
        ```text
        3            # of problem sizes (N)
        152000 160000 168000
        ```
    - The total number of MPI ranks should be NUM_NODES*128. 
    - The process grid should be close to square as much as possible. For example, for 1 node, you can choose P=8, Q=16; for 16 nodes, you can choose P=32, Q=64. 
        ```text
        1            # of process grids (P x Q)
        64           Ps
        128          Qs
        ```
2. Edit `run_pbs.sh` to run jobs at certain node count

3. Submit job: qsub run_pbs.sh

**Note (running from subfolders)**: If you want to run multiple HPL jobs concurrently (different node counts or `HPL.dat` settings), create one subfolder per run and copy `HPL.dat` and `run_pbs.sh` into each subfolder.

When running from a subfolder, update the binary path in `run_pbs.sh` so it points back to the built `xhpl`:
```bash
HPL_BIN=${HPL_BIN:-"../hpl-2.3/build/bin/xhpl"}
```