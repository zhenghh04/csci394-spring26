# Machine Learning

This module introduces machine learning from an HPC point of view using the
PyTorch framework.

The sequence is based on Chapter 19 of the local course textbook:

- training versus inference
- why GPUs are commonly used for training
- a handwritten-digit example
- distributed training with data parallelism

The intent is not to turn the course into a full machine learning class. The
goal is to help students understand how machine learning workloads fit into
high-performance computing practice.

Contents:

- `Essential_ML.md`
  Slide deck source for the machine-learning lecture.
- `01_intro_framework/`
  Device checks, tensors, and the basic training workflow.
- `02_mnist_linear/`
  A simple MNIST classifier with a minimal model.
- `03_mnist_cnn/`
  A convolutional MNIST example closer to the textbook chapter.
- `04_data_loading_and_batches/`
  Batches, shuffling, train/test splits, and input pipeline cost.
- `05_gpu_training_basics/`
  Compare CPU and GPU training behavior on the same model.
- `06_distributed_training/`
  Introduce data parallel training on multiple processes or devices.
- `assignments/01_mnist_scaling/`
  Measure timing and accuracy across batch sizes and devices.
- `assignments/02_distributed_ml/`
  Strong-scaling or efficiency study for distributed training.

Suggested teaching order:

1. `01_intro_framework`
2. `02_mnist_linear`
3. `04_data_loading_and_batches`
4. `03_mnist_cnn`
5. `05_gpu_training_basics`
6. `06_distributed_training`

What students should learn across the whole sequence:

- the difference between training and inference
- why data movement matters in machine learning workloads
- how GPUs accelerate dense tensor operations
- how model quality and runtime are both part of evaluation
- how data parallel training relates to MPI-style thinking
- how the PyTorch framework hides most low-level GPU details

Recommended software:

- Python 3
- PyTorch
- torchvision

Typical installation:

```bash
python3 -m pip install torch torchvision
```

PyTorch is the recommended framework for this module because the course already
uses it in `07_gpu/pytorch/`, and Chapter 19 explicitly points students toward
PyTorch distributed training as a useful comparison point.

To generate a PowerPoint deck from the slide source:

```bash
pandoc Essential_ML.md -t pptx -o CH19_ML_slides.pptx
```
