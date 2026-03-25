# CUDA 2D Laplacian

This example translates the CPU/OpenMP 5-point stencil from
`05_openmp/12_mechanics_laplacian` into CUDA.

Concepts:

- 2D thread-block mapping for a 2D grid
- one CUDA thread computes one interior stencil point
- boundary handling with a guard clause
- host-device memory allocation and copies

Build:

```bash
make
```

This builds:

- `app` from `main.cu`
- `app_cpu` from OpenMP `main.c`

If your driver is older than the CUDA toolkit, build for the exact GPU target:

```bash
make CUDA_ARCH=sm_80
```

Run:

```bash
./app
./app 4096
./app_cpu
./app_cpu 4096
```

For the CPU version, you can set the thread count with:

```bash
OMP_NUM_THREADS=4 ./app_cpu 4096
```

Argument:

- `n` default: `2048`

Teaching note:

- this version is a direct global-memory stencil
- the next optimization step would be shared-memory tiling to reduce repeated neighbor loads
