# OpenMP target offload example

This module shows a minimal OpenMP offload kernel using:
- `#pragma omp target data`
- `#pragma omp target teams distribute parallel for`

The example computes AXPY (`out[i] = a*x[i] + y[i]`) and checks correctness against a host reference.

This folder now includes four versions:
- `target_axpy.c`: GPU offload path using `target` pragmas.
- `parallel_axpy.c`: simple CPU path using `#pragma omp parallel for`.
- `target_matmul.c`: GPU offload path for dense matrix multiplication `C=A*B`.
- `parallel_matmul.c`: CPU path for dense matrix multiplication `C=A*B`.

## What `target` is doing
- `target` moves execution of a code region from the host CPU to an OpenMP target device (typically a GPU), if one is available.
- `target data` controls data movement for arrays:
  - `map(to: x, y)` copies inputs to device memory.
  - `map(from: out)` copies results back to host memory.
- `target teams distribute parallel for` expresses hierarchical parallelism on the device:
  - `teams`: create groups of workers on the device,
  - `distribute`: split loop chunks across teams,
  - `parallel for`: run iterations in parallel within each team.

If no target device is available, OpenMP usually falls back to host execution unless offload is set to mandatory.

## Build
```bash
make
```

Optional compiler offload target flags can be passed via `OFFLOAD_FLAGS`.

## Run
```bash
# target offload version
./target_axpy
./target_axpy 4000000

# host parallel-for version
./parallel_axpy
OMP_NUM_THREADS=8 ./parallel_axpy 4000000

# target offload matrix-multiplication version
./target_matmul
OMP_TARGET_OFFLOAD=MANDATORY ./target_matmul 1024

# host parallel-for matrix-multiplication version
./parallel_matmul
OMP_NUM_THREADS=8 ./parallel_matmul 1024
```

## Notes
- `num_devices=0` means no target device was found; OpenMP typically falls back to host execution.
- To require real offload (and fail otherwise), run with:
```bash
OMP_TARGET_OFFLOAD=MANDATORY ./target_axpy
```
