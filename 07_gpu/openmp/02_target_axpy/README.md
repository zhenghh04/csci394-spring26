# OpenMP Target AXPY

This example is the first complete numerical OpenMP GPU-offload kernel in the
sequence.

Mathematical operation:

```c
out[i] = a * x[i] + y[i]
```

This is the standard AXPY pattern from BLAS level 1.

What this example does:

- initializes two input vectors on the host
- offloads a 1D loop to the target device
- maps input arrays to the device and the output array back to the host
- validates the result against a host reference expression

Concepts:

- `#pragma omp target teams distribute parallel for`
- automatic mapping with `map(to: ...)` and `map(from: ...)`
- one loop iteration per logical work item
- correctness checking after offload

Why this example comes early:

- the loop is simple enough that students can focus on the OpenMP offload
  structure rather than the math
- it provides a clear bridge between:
  a normal CPU `for` loop
  and
  an offloaded loop using `target teams distribute parallel for`

Key pragma in this example:

```c
#pragma omp target teams distribute parallel for map(to : x[0:n], y[0:n]) map(from : out[0:n])
```

How to read it:

- `target`
  Run the region on an OpenMP target device if available.
- `teams`
  Create groups of workers on the device.
- `distribute`
  Split iterations across teams.
- `parallel for`
  Run loop iterations in parallel within each team.
- `map`
  Control which arrays move to or from the device.

What to look for in the output:

- total runtime for the offloaded loop
- `max_abs_err`
  This should be near zero if the offload computation is correct.

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

- How is this loop different from a normal `#pragma omp parallel for`?
- What data is copied to the device, and what data is copied back?
- Why is AXPY a good first offload example?

What students should learn from this example:

- how to offload a single 1D loop with OpenMP
- how array mapping works at a basic level
- how to check that an offloaded numerical result is still correct
