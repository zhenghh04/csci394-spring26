# OpenACC GPU Offload Examples

These examples mirror the OpenMP target-offload sequence as closely as possible,
but use OpenACC directives instead.

Contents:

- `01_acc_hello/`
  Smallest OpenACC offload sanity check.
- `02_parallel_axpy/`
  First OpenACC offloaded 1D loop.
- `03_data_region/`
  Explicit OpenACC data region example.
- `04_parallel_collapse/`
  2D OpenACC loop nest with `collapse(2)`.
- `05_parallel_laplacian/`
  OpenACC offload of a 2D five-point stencil.
- `06_multi_gpu_axpy/`
  OpenACC multi-GPU AXPY with explicit device selection.
- `07_parallel_reduction/`
  Scalar reduction on the accelerator.

Suggested teaching order:

1. `01_acc_hello`
2. `02_parallel_axpy`
3. `03_data_region`
4. `07_parallel_reduction`
5. `04_parallel_collapse`
6. `05_parallel_laplacian`
7. `06_multi_gpu_axpy`

What students should learn across the whole sequence:

- how OpenACC offload differs from plain CPU loops
- how `parallel loop`, `data`, and `reduction` fit together
- how data movement can be automatic or explicit
- how nested loops are handled with `collapse`
- how multi-GPU offload requires explicit partitioning and device selection
- how directive-based OpenACC compares with OpenMP target offload
