# SYCL Intel XPU Examples

These examples introduce Intel XPU programming with SYCL / oneAPI.

They are intended to be the Intel-GPU counterpart to the CUDA, OpenACC, and
OpenMP examples in this module.

Contents:

- `01_sycl_hello/`
  Small SYCL device-selection and kernel-launch sanity check.
- `02_sycl_axpy/`
  Vector update `y = alpha * x + y` using `parallel_for`.
- `03_sycl_reduction/`
  Scalar sum reduction using the SYCL reduction API.
- `04_sycl_matmul/`
  Dense matrix multiplication `C = A * B` using a 2D `parallel_for`.
- `05_sycl_laplacian/`
  2D five-point Laplacian stencil on a rectangular grid.

Suggested teaching order:

1. `01_sycl_hello`
2. `02_sycl_axpy`
3. `03_sycl_reduction`
4. `04_sycl_matmul`
5. `05_sycl_laplacian`

What students should learn:

- how to create a `sycl::queue`
- how to select an Intel GPU if available
- how `parallel_for` launches device work
- how shared allocations can simplify small examples
- how SYCL reduction differs from manual CUDA reduction code
- how a 2D iteration space maps to dense matrix multiplication
- how stencil operators map to GPU work-items in SYCL

Build:

```bash
make
```

Typical compiler:

```bash
make CXX=icpx
```

Typical oneAPI build flags:

```bash
make CXX=icpx CXXFLAGS='-O2 -std=c++17 -fsycl'
```
