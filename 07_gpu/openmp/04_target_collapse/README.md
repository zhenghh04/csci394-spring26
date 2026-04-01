# OpenMP Target Collapse

This example extends OpenMP GPU offload from a 1D loop to a 2D loop nest.

Mathematical operation:

```c
c[i,j] = a[i,j] + b[i,j]
```

The arrays are stored in linear memory, so the code uses an index helper to map
2D coordinates `(i, j)` onto a 1D array.

What this example does:

- allocates three `n x n` matrices on the host
- initializes `a` and `b`
- offloads the nested `i, j` loops to the target device
- uses `collapse(2)` so OpenMP treats the two loops as one larger iteration space
- checks the result against the host expression

Concepts:

- `collapse(2)` on nested loops
- linearized 2D arrays
- `target teams distribute parallel for collapse(2)`
- matrix-style parallelism on the device

Why `collapse(2)` is important:

- without it, OpenMP parallelizes only the outer loop directly
- with it, the `i` and `j` loops are combined into one larger work space
- that often gives better parallel exposure for 2D data-parallel kernels

Key pragma in this example:

```c
#pragma omp target teams distribute parallel for collapse(2) map(to : a[0:nn], b[0:nn]) map(from : c[0:nn])
```

What to look for in the output:

- runtime for the offloaded matrix-add loop
- `max_abs_err`
  should be near zero
- representative output values such as `c[0]` and the center element

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
./app 2048
OMP_TARGET_OFFLOAD=MANDATORY ./app 2048
```

Suggested class discussion:

- Why does a 2D loop nest still live in a 1D memory array?
- What does `collapse(2)` change conceptually?
- How does this compare with a 2D CUDA launch using `blockIdx.x`, `blockIdx.y`, `threadIdx.x`, and `threadIdx.y`?

What students should learn from this example:

- OpenMP can express 2D parallel work without explicitly programming thread indices
- loop-collapsing is a key idea for nested-loop offload
- OpenMP offload keeps the loop structure visible while CUDA makes the mapping more explicit
