# CUDA pi

This example estimates pi on the GPU with the midpoint-rule numerical integral

- `pi = integral_0^1 4 / (1 + x^2) dx`

Concepts:

- 1D CUDA kernel launch
- per-thread work over integration points
- `cudaMalloc` and `cudaMemcpy`
- correctness check against a CPU reference

Build:

```bash
make
```

If your driver is older than the CUDA toolkit, build for the exact GPU target:

```bash
make CUDA_ARCH=sm_80
```

Run:

```bash
./app
./app 1000000 256
```

Arguments:

- `num_intervals` default: `1048576`
- `threads_per_block` default: `256`
