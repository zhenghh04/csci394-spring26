# OpenMP Target Reduction

This example introduces one of the most important patterns in GPU programming:
compute many values in parallel and combine them into one scalar result.

Mathematical operation:

```c
sum = sum_i x[i] * x[i]
```

This is the squared 2-norm without the final square root.

What this example does:

- initializes one input vector on the host
- offloads a 1D loop to the target device
- uses an OpenMP `reduction(+:sum_gpu)` clause to combine partial sums
- checks the GPU result against a host reference sum

Concepts:

- `#pragma omp target teams distribute parallel for reduction(...)`
- scalar reduction on a target device
- combining parallel partial results into one final value
- validating a floating-point reduction against a CPU reference

Why this example matters:

- it is the OpenMP target analogue of the reduction patterns students will see
  in CUDA and OpenACC
- reductions are one of the first cases where GPU programming is more than
  just "one loop iteration gives one output element"
- it helps students see that offloaded kernels can produce scalar outputs, not
  only arrays

Key pragma in this example:

```c
#pragma omp target teams distribute parallel for map(to : x[0:n]) reduction(+ : sum_gpu)
```

How to read it:

- `target`
  Run the region on an OpenMP target device if available.
- `teams distribute parallel for`
  Partition the loop across device teams and workers.
- `reduction(+ : sum_gpu)`
  Give each worker a private partial sum and combine them with `+` at the end.

What to look for in the output:

- total runtime for the offloaded reduction
- `sum_gpu` and `sum_cpu`
  These should match closely.
- `abs_err` and `rel_err`
  These should be very small for a correct run.

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

- Why is a reduction different from elementwise AXPY?
- What does the reduction clause save us from having to write manually?
- Why can floating-point reductions vary slightly across execution orders?

What students should learn from this example:

- OpenMP offload supports scalar reductions directly
- reductions are a core parallel pattern on GPUs
- correctness checks for floating-point sums should allow for tiny roundoff differences
