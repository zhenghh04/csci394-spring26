# OpenMP Target Data Region

This example separates data movement from computation.

Instead of letting one pragma handle everything automatically, it uses explicit
OpenMP target-data operations so students can see the host-to-device and
device-to-host workflow more clearly.

Mathematical operation:

```c
out[i] = x[i] * x[i] + y[i] * y[i]
```

With the chosen initialization, the result should be close to `1.0` for every
element because `sin^2(theta) + cos^2(theta) = 1`.

What this example does:

- allocates arrays on the host
- creates corresponding arrays on the device with `target enter data`
- copies inputs to the device with `target update to`
- runs an offloaded loop on the device
- copies results back with `target update from`
- frees the device copies with `target exit data`

Concepts:

- `target enter data`
- `target update to`
- `target teams distribute parallel for`
- `target update from`
- `target exit data`
- separating data movement time from kernel time

Why this example matters:

- this is the OpenMP offload analogue of explicit `cudaMalloc`, `cudaMemcpy`,
  and `cudaFree`
- it is the right pattern when the same arrays stay on the device across
  multiple kernels
- it helps students reason about transfer overhead instead of treating mapping
  as invisible magic

What to look for in the output:

- `h2d_s`
  time spent moving inputs to the device
- `kernel_s`
  time spent in the offloaded loop
- `d2h_s`
  time spent copying results back
- `total_s`
  combined runtime
- `max_abs_err`
  should be very small for a correct run

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
./app 4000000
OMP_TARGET_OFFLOAD=MANDATORY ./app 4000000
```

Suggested class discussion:

- When is automatic mapping enough, and when do explicit data regions help?
- Why is transfer time important in GPU performance analysis?
- How does this compare to explicit memory management in CUDA?

What students should learn from this example:

- OpenMP offload has both execution directives and data-movement directives
- data transfer cost can be measured separately from compute cost
- explicit data regions are useful for repeated GPU work on the same arrays
