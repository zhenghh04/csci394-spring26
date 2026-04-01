# OpenMP Target Multi-GPU AXPY

This example extends the earlier single-device AXPY examples to multiple target
devices. It partitions the vector into contiguous chunks and assigns one chunk
to each visible OpenMP target device.

Mathematical operation:

```c
out[i] = a * x[i] + y[i]
```

What this example does:

- queries the number of target devices with `omp_get_num_devices()`
- splits the vector into one chunk per device
- creates device data for each chunk using `device(dev)`
- copies each chunk to its assigned device
- launches one offloaded AXPY loop per device
- copies each result chunk back to the host

Concepts:

- multiple target devices in OpenMP
- `device(dev)` clauses
- manual domain decomposition across GPUs
- repeated `target enter data`, `target update`, and `target exit data`

Why this example matters:

- a normal `target` region uses only one default device
- multi-GPU execution requires explicit chunking and device assignment
- this is the OpenMP analogue of manual multi-GPU partitioning in CUDA or MPI+CUDA workflows

What to look for in the output:

- `num_devices`
  how many target devices OpenMP sees
- `chunks_used`
  how many partitions the input vector was split into
- `device_k_range`
  the index range handled by each device
- `max_abs_err`
  should be near zero

Build:

```bash
make
```

Many compilers need explicit offload flags. Examples:

```bash
make OFFLOAD_FLAGS='-fopenmp-targets=nvptx64-nvidia-cuda'
make CC=nvc OFFLOAD_FLAGS='-mp=gpu'
```

Run:

```bash
./app
./app 8000000
OMP_TARGET_OFFLOAD=MANDATORY ./app 8000000
```

Suggested class discussion:

- Why does a single `target` region not automatically use all GPUs?
- What extra logic is needed for multi-GPU work?
- How is this similar to data partitioning in MPI?

What students should learn from this example:

- multi-GPU offload is an algorithmic choice, not an automatic runtime feature
- OpenMP can target different GPUs, but the programmer must partition the data
- correctness checking matters even when the work is split across devices
