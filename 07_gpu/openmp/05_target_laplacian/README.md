# OpenMP Target Laplacian

This example applies OpenMP GPU offload to a 2D five-point stencil, which is a
common kernel in PDE solvers and mechanics codes.

Mathematical operation:

```c
lap[i,j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
```

Only interior points are updated. The boundary is handled by the loop limits:

- `i = 1 .. n-2`
- `j = 1 .. n-2`

What this example does:

- allocates a 2D field `u` and an output array `lap`
- initializes `u`
- offloads the nested stencil loops to the target device
- uses `collapse(2)` to expose the full interior grid as parallel work
- prints representative stencil outputs

Concepts:

- `target teams distribute parallel for collapse(2)`
- stencil computation on a target device
- linearized 2D indexing
- boundary handling through loop limits

Why this example is useful:

- it mirrors the CUDA stencil examples in the same module
- it shows how OpenMP expresses the same algorithm with directive-based offload
- students can compare explicit CUDA thread mapping with directive-based loop offload

What to look for in the output:

- total runtime for the offloaded stencil
- one near-edge value such as `lap[1,1]`
- one interior value such as `lap[center]`

Build:

```bash
make
```

Many compilers need explicit offload flags. Examples:

```bash
make OFFLOAD_FLAGS='-fopenmp-targets=nvptx64-nvidia-cuda'
make CC=nvc OFFLOAD_FLAGS='-mp=gpu'
```

Run:

```bash
./app
./app 4096
OMP_TARGET_OFFLOAD=MANDATORY ./app 4096
```

Suggested class discussion:

- Why are stencil kernels important in scientific computing?
- Why do the loop bounds skip the boundary points?
- What is simpler in OpenMP offload than in CUDA, and what control is lost?

What students should learn from this example:

- OpenMP offload can express realistic stencil kernels compactly
- loop bounds often encode boundary conditions directly
- the same scientific kernel can be written in both directive-based and explicit GPU styles
