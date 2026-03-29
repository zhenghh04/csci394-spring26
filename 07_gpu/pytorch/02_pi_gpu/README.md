# GPU/CPU Pi (PyTorch)

This example estimates pi with a Monte Carlo method on a single device using PyTorch.

## Requirements
- Python 3
- PyTorch build for your device

Install dependencies:
```bash
# CPU or CUDA or MPS
python -m pip install torch torchvision torchaudio

# Intel XPU
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

## Run
```bash
python3 pi_torch.py --device auto --samples 100000000
```

For a matching native CUDA Monte Carlo example, see:

- `../../cuda/15_cuda_monte_carlo_pi/`

Example comparison on an NVIDIA GPU:

```bash
python3 pi_torch.py --device cuda --samples 100000000 --batch 10000000 --seed 1234
../../cuda/15_cuda_monte_carlo_pi/app 100000000 10000000 256 1234
```

Device options:
- `auto` (default): pick the best available device
- `cuda`: NVIDIA GPU
- `xpu`: Intel XPU
- `mps`: Apple Silicon (Metal)
- `cpu`: CPU baseline

## Notes
- This version does not use MPI; it runs on a single device.
- The PyTorch and CUDA Monte Carlo implementations use different random-number
  generators, so exact hit counts may differ slightly.
