# 05 GPU Training Basics: CPU vs GPU for the Same Model

This lesson makes the HPC connection explicit by comparing the same PyTorch
training problem on CPU and GPU.

Suggested experiment:

- use one fixed MNIST model
- run the same number of epochs on CPU and GPU
- report total time, time per epoch, and final accuracy

Learning goals:

- understand why GPUs are often preferred for training
- see that timing requires synchronization when using GPU work
- separate model quality from raw runtime

Suggested outputs:

- device name
- batch size
- epochs
- training time
- accuracy

Suggested run:

```bash
python3 main.py --device cpu --epochs 3 --batch-size 128
python3 main.py --device cuda --epochs 3 --batch-size 128
```

Suggested setup:

```bash
python3 -m pip install torch torchvision
```

Notes:

- keep timing code explicit
- state clearly if a GPU is not available on the current machine
