# SYCL Reduction

This example computes a scalar reduction:

- `sum = sum_i 1 / (i + 1)`

## Self-explanation

This example shows how to reduce many values into one scalar result.

The program:

1. builds an input array on the host
2. allocates one shared scalar for the final sum
3. launches a SYCL reduction kernel
4. compares the device result with a CPU reference

Why this example matters:

- reductions appear everywhere in scientific computing and AI
- reductions are harder than simple elementwise kernels because many work-items
  contribute to one final value
- SYCL provides a reduction API so you do not have to manually write all the
  low-level synchronization logic

How to read the code:

- `sycl::reduction(sum, sycl::plus<double>())`
  - tells SYCL that all work-items will contribute to one scalar sum
- inside the kernel:
  - each work-item contributes `x[i]`
- after the queue finishes:
  - the code compares the SYCL result with a CPU-computed result

This example is useful for discussing:

- reduction patterns in GPU programming
- why directive and high-level models can be simpler than manual CUDA reduction
- numerical correctness checks for scalar outputs

Concepts:

- SYCL reduction API
- shared allocation for the output scalar
- correctness check against a CPU reference

Build:

```bash
make
```

Run:

```bash
./app
./app 1048576
```

Arguments:

1. input length `n`
