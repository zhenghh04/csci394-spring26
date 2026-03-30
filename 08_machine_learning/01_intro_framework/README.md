# 01 Intro PyTorch: Training, Inference, and Device Basics

This lesson introduces the minimum machine-learning vocabulary needed for the
rest of the module and uses the PyTorch framework from the start.

The starter example is a synthetic linear regression. That keeps the first model
close to familiar numerical computing ideas: vectors, weights, bias, prediction,
and mean-squared error.

Students should leave this lesson understanding:

- what a model is
- what a linear regression model is
- the difference between training and inference
- what an epoch and batch are
- how tensors live on CPU or GPU devices
- how to check whether GPU acceleration is available

Suggested content:

- create a tensor on CPU
- move a tensor to GPU if available
- time a simple tensor operation
- show the structure of a basic training loop
- fit a linearly generated dataset with `nn.Linear`
- use mean-squared error as the loss function

Recommended code deliverable:

- `main.py`
  Trains a small linear regression model on synthetic data and reports MSE.

Suggested run:

```bash
python3 main.py
python3 main.py --device cpu --epochs 20 --samples 2048 --input-dim 8
```

Suggested setup:

```bash
python3 -m pip install torch
```

Notes:

- keep the first example extremely small
- avoid datasets in this lesson
- make device selection explicit and readable
- linear regression is a better first fit than classification because the math is simpler
