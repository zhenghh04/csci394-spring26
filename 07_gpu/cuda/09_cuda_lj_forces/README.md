# CUDA Lennard-Jones forces

This example translates the CPU/OpenMP Lennard-Jones force kernel from
`05_openmp/11_materials_lj` into CUDA.

Concepts:

- one CUDA thread computes the total force for one particle
- pairwise all-to-all interaction with a cutoff radius
- host-to-device copies for particle positions
- device-to-host copies for the force arrays

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
./app 4000 256
./app_cpu
./app_cpu 4000
```

For the CPU version, you can set the thread count with:

```bash
OMP_NUM_THREADS=4 ./app_cpu 4000
```

Arguments:

- `n` default: `2000`
- `threads_per_block` default: `256`

Teaching note:

- this is a direct translation for teaching, not an optimized molecular dynamics implementation
- a faster production code would use neighbor lists or cell lists to avoid the full `O(n^2)` interaction loop
