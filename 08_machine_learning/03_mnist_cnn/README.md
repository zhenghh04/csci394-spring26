# 03 MNIST CNN: Convolutional Model for Handwritten Digits

This lesson follows the spirit of Chapter 19 more closely by using a
convolutional neural network for MNIST in PyTorch.

Suggested model ingredients:

- convolution
- pooling
- flatten
- dropout
- final dense classification layer

Learning goals:

- see why convolutional layers are useful for images
- compare a CNN against the simpler linear baseline
- observe the tradeoff between extra computation and improved accuracy

Suggested outputs:

- training loss and accuracy
- validation or test accuracy
- total training time
- model parameter count if convenient

Suggested run:

```bash
python3 main.py --epochs 5 --batch-size 128 --device auto
```

Suggested setup:

```bash
python3 -m pip install torch torchvision
```

Notes:

- this lesson should connect directly to the textbook MNIST example
- keep the architecture modest so runtime stays reasonable on student hardware
