# CUDA threads and blocks

This is the first minimal CUDA kernel example for class.

Concepts:

- `blockIdx.x`
- `threadIdx.x`
- `blockDim.x`
- global index calculation
- launch configuration with `<<<num_blocks, threads_per_block>>>`

Build:

```bash
make
```

If your driver is older than the CUDA toolkit, build for the exact GPU target:

```bash
make CUDA_ARCH=sm_80
```

Common examples:

- `sm_80` for A100
- `sm_90` for H100
- `sm_86` for many RTX 30-series cards

Run:

```bash
./app
./app 20 6
```

Arguments:

- first argument: `N`, the number of logical work items
- second argument: `threads_per_block`

The program launches a 1D CUDA kernel, stores each thread's block and thread IDs
into arrays, copies them back to the host, and prints a table showing how the
work is mapped onto blocks and threads.
