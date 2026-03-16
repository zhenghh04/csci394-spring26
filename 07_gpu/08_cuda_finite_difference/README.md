# CUDA finite difference

This Chapter 18 example computes the x-derivative of a 3D field on the GPU using
an explicit CUDA kernel.

Concepts:

- explicit CUDA kernel launch
- grid and block dimensions
- GPU global memory allocation
- host-to-device and device-to-host copies
- accuracy check against `cos(x)`

Build:

```bash
make
```

Run:

```bash
./app
./app 128
```
