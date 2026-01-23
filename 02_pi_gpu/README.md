# GPU Pi Examples

These examples estimate pi with a Monte Carlo method on a single GPU.

## NVIDIA GPU (PyTorch CUDA)
### Requirements
- Python 3
- PyTorch CUDA build
- NVIDIA GPU with CUDA drivers

Install dependencies:
```bash
python -m pip install torch torchvision torchaudio
```

### Run
```bash
python3 pi_torch_cuda.py --samples 100000000
```

## Intel XPU (PyTorch)
### Requirements
- Python 3
- PyTorch XPU build
- Intel GPU with drivers + oneAPI runtime

Install dependencies:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### Run
```bash
python3 pi_torch_xpu.py --samples 100000000
```

## Apple Silicon (Metal / MPS)
### Requirements
- Python 3
- PyTorch with MPS support
- macOS with Apple Silicon

Install dependencies:
```bash
python -m pip install torch torchvision torchaudio
```

### Run
```bash
python3 pi_torch_mps.py --samples 100000000
```

## CPU baseline (PyTorch)
### Requirements
- Python 3
- PyTorch (CPU build)

Install dependencies:
```bash
python -m pip install torch torchvision torchaudio
```

### Run
```bash
python3 pi_torch_cpu.py --samples 100000000
```

## Notes
- These versions do not use MPI; they run on a single device (CPU or GPU).
