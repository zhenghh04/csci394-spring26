# OpenMP target offload example

This module shows a minimal OpenMP offload kernel using:
- `#pragma omp target data`
- `#pragma omp target teams distribute parallel for`

The example computes AXPY (`out[i] = a*x[i] + y[i]`) and checks correctness against a host reference.

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
./target_axpy
./target_axpy 4000000
```

## Notes
- `num_devices=0` means no target device was found; OpenMP typically falls back to host execution.
- To require real offload (and fail otherwise), run with:
```bash
OMP_TARGET_OFFLOAD=MANDATORY ./target_axpy
```
