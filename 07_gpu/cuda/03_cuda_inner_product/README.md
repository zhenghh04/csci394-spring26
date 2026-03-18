# CUDA inner product

This example computes a vector inner product on the GPU with a simple explicit
CUDA kernel.

Concepts:

- 1D kernel launch
- global index calculation
- `cudaMalloc` and `cudaMemcpy`
- simple GPU elementwise multiply
- correctness check against a CPU reference
- CPU vs GPU timing comparison

Teaching note:

- this version keeps the reduction simple for class discussion
- the GPU computes the elementwise products
- the host sums the temporary product array
- timing separates CPU time from GPU transfer and kernel time

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
./app 1000000
```
