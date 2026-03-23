# CUDA pi

This example estimates pi on the GPU with the midpoint-rule numerical integral

- `pi = integral_0^1 4 / (1 + x^2) dx`

Concepts:

- 1D CUDA kernel launch
- per-thread work over integration points
- GPU reduction so only the final sum is copied back to the host
- `cudaMalloc` and minimal `cudaMemcpy`
- correctness check against a CPU reference
- CPU and GPU timing comparison with reported speedup

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
- `threads_per_block` should be a power of two for the reduction
