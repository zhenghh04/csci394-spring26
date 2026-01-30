# HPL (High Performance Linpack)

HPL is typically provided by the supercomputer as an optimized binary. This
folder includes a template `HPL.dat` and a sample PBS script for running HPL.

## Steps (typical)
1. Load the vendor or cluster HPL module (or build HPL if required).
2. Edit `HPL.dat` to match the number of MPI ranks and problem sizes you want.
3. Submit `run_pbs.sh`.

## Notes
- The `P x Q` process grid in `HPL.dat` must multiply to the number of ranks.
- Problem size `N` should be large enough to fill most of the node memory.
- Use a block size (`NB`) that matches your cluster guidance (often 192–384).

If your cluster uses a different binary name or launch method, update
`run_pbs.sh` accordingly.
