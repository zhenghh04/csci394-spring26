# SYCL AXPY

This example computes:

- `y = alpha * x + y`

## Self-explanation

This is one of the standard beginner GPU kernels.

The program:

1. allocates vectors `x` and `y`
2. initializes them on the host
3. launches a SYCL `parallel_for`
4. computes one output element per work-item
5. compares the result with a CPU reference

Why this example matters:

- AXPY is simple enough to understand immediately
- the operation is parallel because each element can be updated independently
- it demonstrates the SYCL pattern of mapping array work onto many device
  work-items
- it provides a direct bridge to similar examples in CUDA, OpenACC, OpenMP, and
  PyTorch

How to read the code:

- `x` and `y` are allocated with `malloc_shared`
  - this avoids explicit copy calls in a small teaching example
- `parallel_for(range<1>(n), ...)`
  - launches `n` independent work-items
- inside the kernel:
  - `y[i] = alpha * x[i] + y[i]`
- after `q.wait()`:
  - the host checks `max_abs_err` against a CPU-computed reference

This example is useful for discussing:

- data parallelism
- correctness checks
- kernel launch overhead versus useful work
- how high-level elementwise operations map to GPU execution

Concepts:

- `parallel_for`
- shared memory allocation with `malloc_shared`
- correctness check against a CPU reference
- simple timing with queue synchronization

Build:

```bash
make
```

Run:

```bash
./app
./app 1048576 2.0
```

Arguments:

1. vector length `n`
2. scalar `alpha`
