# SYCL Laplacian

This example applies the 2D five-point Laplacian operator on a grid:

- `out[i, j] = -4 * u[i, j] + u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]`

for interior points.

## Self-explanation

This example introduces a stencil computation in SYCL.

The program:

1. allocates 2D arrays stored in 1D row-major form
2. initializes the input grid
3. launches a 2D SYCL kernel
4. computes the five-point Laplacian for interior points
5. compares the device result with a CPU reference

Why this example matters:

- stencil operators are common in PDEs, image processing, and scientific
  computing
- unlike AXPY, each output depends on neighboring grid points
- it helps students reason about memory access patterns and boundary handling
- it provides a direct SYCL counterpart to the OpenACC, OpenMP, and CUDA
  Laplacian examples in this module

How to read the code:

- `parallel_for(range<2>(ny, nx), ...)`
  - launches one work-item for each grid point
- inside the kernel
  - boundary points are set to zero
  - interior points read the center and four neighbors
- after `q.wait()`
  - the host computes a CPU reference and reports `max_abs_err`

This is a simple teaching implementation, not an optimized shared-memory tile
version.

## Concepts

- 2D stencil computation
- 2D `parallel_for`
- row-major indexing for grids
- boundary-condition handling
- correctness check against a CPU reference

## Build

```bash
make
```

## Run

```bash
./app
./app 1024 1024
```

Arguments:

1. grid size in `x`, `nx`
2. grid size in `y`, `ny`
