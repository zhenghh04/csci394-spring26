# HPCG (High Performance Conjugate Gradients)

HPCG is a complementary benchmark to HPL, focusing on memory bandwidth and
sparse matrix performance. It models a 3D diffusion problem solved by a
preconditioned conjugate gradient (PCG) method, and is more representative of
many real applications than dense linear algebra. Many clusters provide an
optimized `xhpcg` binary as a module. This folder includes a template
`hpcg.dat` and a PBS script to run it.

## What HPCG measures
- Sparse matrix-vector and triangular solves (memory bandwidth bound)
- Global reductions and halo exchanges (communication effects)
- Overall sustained GFLOP/s for a fixed algorithmic workload

HPCG typically reports:
- **GFLOP/s** (the main figure of merit)
- Problem size (`Nx Ny Nz`)
- Total ranks and the `P x Q x R` process grid
- Runtime and iteration statistics

## Typical steps
1. Load the HPCG module (or build it if required by your system).
2. Edit `hpcg.dat` to set the global problem size and runtime.
3. Submit `run_pbs.sh`.

## Build (optional, from source)
If your cluster does not provide `xhpcg`, you can build from source:
```bash
./build_hpcg.sh
```
The script downloads the official tarball, creates a minimal `Make.<ARCH>`, and
builds `bin/<ARCH>/xhpcg`. Override variables as needed:
```bash
HPCG_VERSION=3.1 ARCH=Linux MPICXX=mpicxx MPICC=mpicc ./build_hpcg.sh
```

### OpenMP note (macOS / clang)
If you see:
```
fatal error: 'omp.h' file not found
```
install OpenMP and rebuild with OpenMP flags:
```bash
brew install libomp
MPICXX=mpicxx MPICC=mpicc ARCH=Linux \
  CXXFLAGS="-O3 -std=c++11 -fopenmp" \
  LDFLAGS="-fopenmp" \
  ./build_hpcg.sh
```

## Notes
- `hpcg.dat` uses a 3D grid size: `Nx Ny Nz` (global problem size).
- The grid should be large enough to fill memory but still fit on the nodes.
- The number of MPI ranks should match your `P x Q x R` process grid.
- HPCG is sensitive to memory bandwidth and MPI latency; results vary across systems.

## Choosing a problem size
Rules of thumb:
- Start small (e.g., `64 64 64`) to validate the run.
- Increase size until each rank has enough work (often tens of millions of unknowns
  total across the job).
- If you see very low GFLOP/s, the problem may be too small or communication-bound.

## Output files
HPCG writes an output file such as `HPCG-Benchmark_*.txt` in the run directory.
That file contains the final GFLOP/s line and detailed configuration info.

If your site uses a different binary name or launch method, update
`run_pbs.sh` accordingly.
