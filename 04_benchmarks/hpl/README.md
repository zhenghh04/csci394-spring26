# HPL (High Performance Linpack)

HPL is typically provided by the supercomputer as an optimized binary. This
folder includes a template `HPL.dat` and a sample PBS script for running HPL.

## Build (from source with provided script)
This folder includes a helper script that downloads HPL from Netlib and builds
it with your MPI compiler and BLAS. You may need to load modules first.

```bash
# Example: load MPI + BLAS (adjust for your cluster)
# module purge
# module load mpi
# module load openblas

# Build HPL (downloads and compiles)
make
```

### Build customization (common)
You can override these environment variables when building:
- `HPL_VERSION` (default 2.3)
- `HPL_URL` (if your site mirrors HPL)
- `CC` / `LINKER` (MPI compiler wrappers)
- `BLAS_LIBS` (e.g., `-lopenblas` or MKL link line)
- `ARCH` (architecture name used by HPL, default `UNKNOWN`)

Example with MKL:
```bash
BLAS_LIBS='-L/path/to/mkl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm' make
```

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
