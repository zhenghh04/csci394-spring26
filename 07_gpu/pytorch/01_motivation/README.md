# 00 Motivation: Simple CPU vs GPU Matrix Multiplication with PyTorch

This example is intentionally very small.

It allocates two square matrices, computes `C = A @ B` on the CPU, and then
does the same on the GPU if CUDA is available.

## Requirements

- Python 3
- PyTorch
- CUDA-enabled PyTorch build if you want GPU results

Example install:

```bash
python3 -m pip install torch
```

## Run

```bash
python3 main.py
```

## Notes

- The code is meant to be easy to read in class, not a full benchmarking framework.
- Change `matrix_size` at the top of `main.py` if you want a different problem size.
- CUDA timings use synchronization so the GPU time includes the real matrix multiplication.
- If CUDA is not available, the script prints a CPU baseline only.
- Large matrices can consume a lot of memory. Reduce `matrix_size` if you hit an out-of-memory error.
