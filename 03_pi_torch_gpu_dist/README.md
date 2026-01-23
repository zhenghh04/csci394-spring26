# Distributed Pi (PyTorch + MPI)

This example estimates pi with a Monte Carlo method using PyTorch and `torch.distributed`. It runs one rank per process and can use CPU, CUDA, XPU, or MPS depending on what's available.

## Requirements
- Python 3
- PyTorch build that supports your device (CUDA/XPU/MPS/CPU)
- MPI runtime (Open MPI, MPICH, etc.)

## Run
```bash
mpiexec -n 4 python3 pi_torch_mpi_dist.py --samples 100000000
```

## Notes
- `--samples` is the total number of random points across all ranks.
- Each rank uses a different random seed.
- If multiple GPUs are available, ranks are mapped round-robin to devices.
- Backend selection: `nccl` for CUDA, `ccl` for Intel XPU, `gloo` for CPU/MPS.
