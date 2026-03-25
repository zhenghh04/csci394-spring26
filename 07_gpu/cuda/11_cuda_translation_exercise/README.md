# CUDA translation exercise

This in-class exercise asks students to convert a simple CPU stencil into a CUDA program.

Files:

- `main.c` - complete CPU baseline
- `main.cu` - CUDA starter with `TODO` markers
- `solution.cu` - instructor solution

Kernel to translate:

```c
out[i] = in[i - 1] + 2.0 * in[i] + in[i + 1];
```

Concepts:

- 1D CUDA kernel launch
- global index calculation
- boundary guards
- `cudaMalloc`
- `cudaMemcpy`
- launch configuration with `<<<num_blocks, threads_per_block>>>`

Suggested in-class workflow:

1. Build and run the CPU baseline.
2. Ask students to identify the loop body that becomes the CUDA kernel.
3. Have students fill in the `TODO` sections in `main.cu`.
4. Compare with `solution.cu`.

Build:

```bash
make app_cpu
make app_solution
```

If students complete `main.cu`, they can build it with:

```bash
make app
```

If your driver is older than the CUDA toolkit, build for the exact GPU target:

```bash
make app_solution CUDA_ARCH=sm_80
```

Run:

```bash
./app_cpu
./app_solution
./app_solution 1000000 256
```

Notes for teaching:

- the exercise is intentionally 1D so students can focus on the CUDA workflow
- it is simpler than the later stencil examples in `10_cuda_laplacian`
- the solution uses a direct global-memory implementation with no optimization
