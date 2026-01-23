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

Device options:
- `auto` (default): pick the best available device
- `cuda`: NVIDIA GPU
- `xpu`: Intel XPU
- `mps`: Apple Silicon (Metal)
- `cpu`: CPU baseline

## Notes
- This version does not use MPI; it runs on a single device.
