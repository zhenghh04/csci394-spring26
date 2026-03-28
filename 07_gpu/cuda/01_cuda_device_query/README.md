# CUDA device query

This example is about the first CUDA runtime calls you should make on a new
machine. It checks whether the CUDA runtime can see any NVIDIA GPUs and then
prints basic properties of device `0`.

In class, this is the right "hello CUDA" step before launching kernels. If
this program fails, the later CUDA examples will fail too, so it acts as a
sanity check for the software stack and the visible hardware.

Prerequisites:

- NVIDIA driver installed
- CUDA toolkit available on `PATH` so `nvcc` is found by `make`

Concepts:

- `cudaGetDeviceCount`
- `cudaGetDeviceProperties`
- querying device metadata from the CUDA runtime
- quick sanity check before running larger CUDA examples

What the program does:

1. Calls `cudaGetDeviceCount` to ask the runtime how many CUDA-capable devices
   are available.
2. Stops with an error message if CUDA is unavailable or no GPU is visible.
3. Calls `cudaGetDeviceProperties` for device `0`.
4. Prints a few important hardware properties that matter for later examples.

Reported fields:

- `device_count`: how many CUDA-capable GPUs the runtime can see
- `device_0_name`: the model name of the first visible GPU
- `compute_capability`: the GPU architecture version as `major.minor`
- `global_mem_gb`: total global memory on that GPU
- `multi_processor_count`: number of streaming multiprocessors (SMs)
- `max_threads_per_block`: largest block size allowed for one kernel launch

Why these fields matter:

- compute capability helps you choose an explicit `CUDA_ARCH` value if needed
- global memory limits the problem sizes you can fit on the device
- SM count gives a rough idea of how much parallel hardware is available
- maximum threads per block constrains launch configurations in later kernels

Build:

```bash
make
```

If your driver is older than the CUDA toolkit, build for the exact GPU target:

```bash
make CUDA_ARCH=sm_80
```

`CUDA_ARCH` reference:

| `CUDA_ARCH` | Compute capability | NVIDIA GPU generation | Example GPUs |
| --- | --- | --- | --- |
| `sm_70` | 7.0 | Volta | V100 |
| `sm_75` | 7.5 | Turing | T4, RTX 2080 |
| `sm_80` | 8.0 | Ampere datacenter | A100 |
| `sm_86` | 8.6 | Ampere client/workstation | RTX 3090, RTX A6000 |
| `sm_89` | 8.9 | Ada Lovelace | RTX 4090, L4 |
| `sm_90` | 9.0 | Hopper | H100 |

Use the `compute_capability` value printed by this example to choose the
matching `CUDA_ARCH` when a manual target is needed.

Run:

```bash
./app
```

Example output:

```text
CUDA device query
device_count=1
device_0_name=NVIDIA A100-SXM4-40GB
compute_capability=8.0
global_mem_gb=40.00
multi_processor_count=108
max_threads_per_block=1024
```

Typical use:

- run this example first to verify the machine can see a CUDA device
- note the reported compute capability before picking a manual `CUDA_ARCH`
  value in later examples
- compare `max_threads_per_block` with the thread-block sizes used in later
  kernel launch examples
