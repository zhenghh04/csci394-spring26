# CUDA data movement bandwidth

This example measures host-to-device and device-to-host bandwidth using
`cudaMemcpy`.

Concepts:

- pinned host memory with `cudaMallocHost`
- device memory allocation with `cudaMalloc`
- timing with CUDA events
- effective transfer bandwidth in GB/s

Teaching note:

- this example isolates transfer cost rather than kernel cost
- it helps students see why data movement can dominate total runtime

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
./app 67108864 20
```

Arguments:

- first argument: number of `float` values
- second argument: number of timed iterations
