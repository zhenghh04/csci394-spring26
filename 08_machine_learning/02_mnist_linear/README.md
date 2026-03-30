# 02 MNIST Linear: First End-to-End Classifier

This lesson uses the MNIST handwritten-digit dataset to build the first complete
PyTorch training workflow in the course.

The model should be intentionally simple:

- flatten the image
- apply one or two linear layers
- train for a small number of epochs

Learning goals:

- load a standard dataset
- separate training and test data
- compute loss and accuracy
- understand why a simple baseline matters before using a larger model

Suggested outputs:

- training loss per epoch
- test accuracy
- total training time

Suggested run:

```bash
python3 main.py --epochs 5 --batch-size 128 --device auto
```

Suggested setup:

```bash
python3 -m pip install torch torchvision
```

Notes:

- this lesson should emphasize clarity over accuracy
- keep the model small enough that students can read the whole file in class
