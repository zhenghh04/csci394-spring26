# cuBLAS GEMM

This example multiplies two small dense matrices on the GPU with
`cublasSgemm`.

Concepts:

- cuBLAS handle creation
- GPU matrix allocation
- host-to-device and device-to-host copies
- `cublasSgemm` for `C = alpha * A * B + beta * C`
- column-major storage convention used by BLAS libraries
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
```
