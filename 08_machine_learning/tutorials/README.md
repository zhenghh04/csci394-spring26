# Deep Learning Tutorials

**CSCI 394 -- Spring 2026**

This directory contains three hands-on tutorials that progressively introduce
deep learning for image classification using PyTorch. Each tutorial is available
as both a Jupyter notebook (for Google Colab) and a standalone Python script.

| # | Tutorial | Notebook | Script |
| - | -------- | -------- | ------ |
| 1 | MNIST with Fully Connected Network | `01_mnist_deep_learning.ipynb` | `01_mnist_deep_learning.py` |
| 2 | CIFAR-10 with Fully Connected Network | `02_cifar10_deep_learning.ipynb` | `02_cifar10_deep_learning.py` |
| 3 | CIFAR-10 with CNN | `03_cifar10_cnn.ipynb` | `03_cifar10_cnn.py` |

Training results and comparison plots are in `figures/` and summarized in
`report.md`.

---

## 1. What is Deep Learning?

Deep learning is a subset of machine learning that uses **neural networks**
with multiple layers to learn representations of data. Instead of manually
designing features (e.g., edge detectors, color histograms), the network
learns the right features directly from raw data.

```
Traditional ML:          Deep Learning:

  Raw Data                 Raw Data
     |                        |
     v                        v
 Hand-crafted           +-----------+
  Features              | Layer 1   |  <- learns low-level features
     |                  +-----------+
     v                        |
 +----------+           +-----------+
 | Classifier|          | Layer 2   |  <- learns mid-level features
 +----------+           +-----------+
     |                        |
     v                  +-----------+
  Prediction            | Layer 3   |  <- learns high-level features
                        +-----------+
                              |
                              v
                         Prediction
```

### Why "deep"?

The term refers to the **depth** of the network -- the number of layers
between input and output. Deeper networks can learn more abstract,
hierarchical representations.

---

## 2. The Artificial Neuron

Every neural network is built from **neurons** (also called units or nodes).
A single neuron computes a weighted sum of its inputs, adds a bias, and
passes the result through an **activation function**:

```
  Inputs        Weights
                                         Activation
  x_1 ---w_1--\                          Function
                \                            |
  x_2 ---w_2----[  sum + bias = z  ] ----> f(z) ----> output
                /
  x_3 ---w_3--/

  z = w_1*x_1 + w_2*x_2 + w_3*x_3 + b
  output = f(z)
```

### Common Activation Functions

```
  ReLU: f(z) = max(0, z)         Sigmoid: f(z) = 1/(1+e^(-z))

  output                          output
    |      /                        |        ------
    |     /                         |      /
    |    /                          |    /
  --+---/--------> z              --+--/----------> z
    |  0                            | 0
    |                               |
```

**ReLU** (Rectified Linear Unit) is the most widely used activation function
in modern deep learning because it is simple, fast, and avoids the
vanishing gradient problem.

---

## 3. Fully Connected (Dense) Layers

In a **fully connected** (FC) layer, every neuron in one layer is connected
to every neuron in the next layer. This is also called a "dense" layer.

### Architecture for MNIST (Tutorial 1)

MNIST images are 28x28 = 784 pixels. We flatten them into a vector and
pass through FC layers:

```
  Input Image          Flatten         Hidden Layer 1    Hidden Layer 2    Output
   28 x 28            784 neurons       256 neurons       128 neurons     10 neurons
  +--------+         +---+             +---+             +---+           +---+
  |        |         | . |----\   /--->| . |----\   /--->| . |---\  /--->| 0 |
  |   7    |  --->   | . |-----\-/-+-->| . |-----\-/-+-->| . |----\-/-+->| 1 |
  |        |         | . |------X--+-->| . |------X--+-->| . |-----X--+->| 2 |
  +--------+         | . |-----/-\-+-->| . |-----/-\-+-->| . |----/-\-+->| . |
                      | . |----/   \-->| . |----/   \-->| . |---/  \--->| . |
                      +---+             +---+             +---+           | 9 |
                                       + ReLU            + ReLU          +---+
                                                                        Softmax
                      784               256               128             10
                         \___________/     \___________/     \_________/
                          784x256+256       256x128+128       128x10+10
                          = 200,960         = 32,896          = 1,290
                                                          Total: 235,146
```

Each arrow represents a **learnable weight**. The total number of parameters
in a layer is: `(input_size x output_size) + output_size` (weights + biases).

### How Prediction Works

The output layer has 10 neurons (one per digit). The **softmax** function
converts the raw outputs (logits) into probabilities:

```
  Logits:    [ 0.1,  8.3,  0.5,  0.2,  0.0,  0.3,  0.1,  0.4,  0.0,  0.1 ]
                      ^
                   highest
                      |
                      v
  Softmax:   [ 0.0,  0.98, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ]
                      ^
              Predicted class: 1
```

---

## 4. How Neural Networks Learn

Training a neural network involves four steps repeated over many iterations:

```
  +------------------+     +------------------+     +------------------+
  | 1. FORWARD PASS  | --> | 2. COMPUTE LOSS  | --> | 3. BACKWARD PASS |
  |                  |     |                  |     |  (Backpropagation)|
  | Feed input       |     | Compare output   |     | Compute gradients|
  | through layers   |     | to true label    |     | for every weight |
  +------------------+     +------------------+     +------------------+
           ^                                                  |
           |                                                  v
           |                                        +------------------+
           +--------------------------------------- | 4. UPDATE WEIGHTS|
                                                    |                  |
                                                    | w = w - lr * grad|
                                                    +------------------+
```

### Step-by-step

1. **Forward pass**: Input flows through each layer, producing a prediction.

2. **Loss computation**: A loss function measures how wrong the prediction is.
   - For classification: **Cross-Entropy Loss**
   - For regression: **Mean Squared Error (MSE)**

3. **Backpropagation**: Using the chain rule of calculus, compute how much
   each weight contributed to the error (the gradient).

4. **Weight update**: Adjust each weight in the direction that reduces the
   loss, scaled by a **learning rate** (lr).

### Key Terminology

```
  Epoch:       One complete pass through the entire training dataset
  Batch:       A subset of training samples processed together
  Batch size:  Number of samples in one batch (e.g., 128)
  Iteration:   One weight update (= one batch processed)

  Example: 60,000 samples, batch size 128
           -> 469 iterations per epoch
```

### Loss Landscape (Conceptual)

```
  Loss
   ^
   |  \
   |   \       /\
   |    \     /  \     Local
   |     \   /    \    minimum
   |      \_/      \        /\
   |                \      /  \
   |                 \    /
   |                  \  /
   |                   \/   <-- Global minimum (goal)
   +----------------------------> Weight values
```

The optimizer navigates this landscape, trying to find the lowest point (minimum loss).

---

## 5. Convolutional Neural Networks (CNNs)

### The Problem with FC Networks for Images

Fully connected networks have fundamental limitations for image data:

```
  28x28 image -> Flatten -> 784 inputs to FC layer

  Problem 1: Destroys spatial structure
  +--------+                    +---+
  | pixels |  -> flatten ->     | . |  No concept of
  | have   |                    | . |  "which pixels
  | spatial|                    | . |  are neighbors"
  | meaning|                    | . |
  +--------+                    +---+

  Problem 2: No translation invariance
  +--------+    +--------+
  |  7     |    |     7  |     FC treats these as
  |        | != |        |     completely different
  +--------+    +--------+     inputs

  Problem 3: Too many parameters
  32x32x3 = 3072 inputs x 512 neurons = 1,572,864 weights (first layer alone!)
```

### The Convolution Operation

A **convolution** slides a small filter (kernel) across the image, computing
a dot product at each position. This detects **local patterns** like edges.

```
  Input (5x5)              Filter (3x3)           Output (3x3)
  +---+---+---+---+---+   +---+---+---+
  | 1 | 0 | 1 | 0 | 1 |   | 1 | 0 | 1 |          +---+---+---+
  +---+---+---+---+---+   +---+---+---+          | 4 | 3 | 4 |
  | 0 | 1 | 0 | 1 | 0 |   | 0 | 1 | 0 |   --->   +---+---+---+
  +---+---+---+---+---+   +---+---+---+          | 2 | 4 | 3 |
  | 1 | 0 | 1 | 0 | 1 |   | 1 | 0 | 1 |          +---+---+---+
  +---+---+---+---+---+                           | 4 | 3 | 4 |
  | 0 | 1 | 0 | 1 | 0 |                           +---+---+---+
  +---+---+---+---+---+
  | 1 | 0 | 1 | 0 | 1 |   Computation at top-left:
  +---+---+---+---+---+   1*1 + 0*0 + 1*1
                           + 0*0 + 1*1 + 0*0
                           + 1*1 + 0*0 + 1*1 = 4
```

### Key Properties of Convolutions

**Parameter sharing**: The same 3x3 filter (9 weights) is applied at
every spatial position. Compare this to FC where each position has its
own set of weights.

```
  FC layer (5x5 -> 3x3):   5*5 * 3*3 = 225 weights
  Conv layer (3x3 filter):  3*3       =   9 weights  (25x fewer!)
```

**Translation invariance**: Since the same filter scans everywhere, a
pattern is detected regardless of where it appears in the image.

### Max Pooling

**Max pooling** reduces the spatial dimensions by taking the maximum value
in each non-overlapping patch:

```
  Input (4x4)                        Output (2x2)
  +----+----+----+----+              +----+----+
  |  1 |  3 |  2 |  1 |              |    |    |
  +----+----+----+----+    2x2       |  4 |  3 |
  |  4 |  2 |  1 |  3 |   MaxPool   |    |    |
  +----+----+----+----+  -------->   +----+----+
  |  1 |  0 |  3 |  2 |              |    |    |
  +----+----+----+----+              |  2 |  4 |
  |  2 |  1 |  1 |  4 |              |    |    |
  +----+----+----+----+              +----+----+

  max(1,3,4,2)=4   max(2,1,1,3)=3
  max(1,0,2,1)=2   max(3,2,1,4)=4
```

Pooling serves two purposes:
- **Reduces computation** by shrinking feature maps
- **Adds invariance** to small spatial shifts

### Batch Normalization

Normalizes activations within each mini-batch to have zero mean and unit
variance per channel. This stabilizes training and allows higher learning
rates.

```
  Without BatchNorm:                With BatchNorm:
  Layer activations                 Layer activations
  can drift wildly                  are kept centered
  during training.                  and scaled.

  |    *                            |   *
  |  *   *                          |  * *
  | *  *   *     unstable           | * * *   stable
  +---*--------->                   +--*-*---------->
```

### CNN Architecture (Tutorial 3)

Our CIFAR-10 CNN has three convolutional blocks followed by a classifier:

```
  Input: 3 x 32 x 32 (RGB image)
  |
  |  Block 1
  |  +--Conv2d(3->32, 3x3, pad=1)--BatchNorm--ReLU--+
  |  +--Conv2d(32->32, 3x3, pad=1)-BatchNorm--ReLU--+
  |  +--MaxPool2d(2x2)------------------------------+
  |                    32 x 16 x 16
  |  Block 2
  |  +--Conv2d(32->64, 3x3, pad=1)--BatchNorm--ReLU--+
  |  +--Conv2d(64->64, 3x3, pad=1)--BatchNorm--ReLU--+
  |  +--MaxPool2d(2x2)-------------------------------+
  |                    64 x 8 x 8
  |  Block 3
  |  +--Conv2d(64->128, 3x3, pad=1)-BatchNorm--ReLU--+
  |  +--Conv2d(128->128, 3x3, pad=1)-BatchNorm--ReLU-+
  |  +--MaxPool2d(2x2)-------------------------------+
  |                    128 x 4 x 4
  |
  |  Classifier
  |  +--Flatten (128*4*4 = 2048)---+
  |  +--Dropout(0.5)---------------+
  |  +--Linear(2048, 256)--ReLU----+
  |  +--Dropout(0.3)---------------+
  |  +--Linear(256, 10)------------+
  |
  v
  Output: 10 class probabilities
```

### What Each Layer Learns

As data flows through the CNN, each block extracts increasingly abstract features:

```
  Block 1 (early layers):       Block 2 (middle layers):     Block 3 (deep layers):
  Low-level features            Mid-level features           High-level features

  +-------+  +-------+         +-------+  +-------+         +-------+
  | / / / |  | - - - |         |  ___  |  | /\_/\ |         | (cat) |
  | / / / |  | - - - |         | /   \ |  | \   / |         |  face |
  | / / / |  | - - - |         | \___/ |  |  \_/  |         |       |
  +-------+  +-------+         +-------+  +-------+         +-------+
   edges,     horizontal        curves,     textures,         object
   gradients  lines             circles     patterns          parts
```

---

## 6. FC vs CNN: Side-by-Side Comparison

```
  Fully Connected Network              Convolutional Neural Network
  =========================            ============================

  Input                                Input
  [flatten to 1D vector]               [keep 2D spatial structure]
       |                                    |
  +----------+                         +----------+
  | FC Layer | every pixel             | Conv 3x3 | local 3x3
  | 3072x512 | connects to            | 3->32    | patch only
  | = 1.5M   | every neuron           | = 864    | (shared weights)
  | weights   |                        | weights  |
  +----------+                         +----------+
       |                                    |
  +----------+                         +----------+
  | FC Layer |                         | MaxPool  | shrink spatial
  +----------+                         +----------+
       |                                    |
  +----------+                         +----------+
  | FC Layer |                         | Conv 3x3 | deeper features
  +----------+                         +----------+
       |                                    |
    Output                             +----------+
                                       | FC Layer | small input
                                       +----------+ (after pooling)
                                            |
                                         Output

  Parameters: ~1,739,000                Parameters: ~815,000
  CIFAR-10 accuracy: ~55%               CIFAR-10 accuracy: ~87%
```

### Why CNNs Win

| Property | FC Network | CNN |
| -------- | ---------- | --- |
| Spatial structure | Destroyed by flattening | Preserved by 2D convolutions |
| Parameter sharing | None (unique weights per position) | Same filter applied everywhere |
| Translation invariance | None | Built-in via convolution + pooling |
| Parameter count | Grows with image size squared | Independent of image size |
| Hierarchical features | Flat (single transformation) | Edges -> Textures -> Objects |

---

## 7. Training Techniques

### Data Augmentation

Artificially expand the training set by applying random transformations:

```
  Original         RandomCrop        HorizontalFlip      Both
  +--------+       +--------+        +--------+          +--------+
  |  cat   |       |   cat  |        |  tac   |          |   tac  |
  |  >^.^< |  -->  |  >^.^  |  or   |  >^.^< |    or   |  ^.^<  |
  |        |       |        |        |        |          |        |
  +--------+       +--------+        +--------+          +--------+

  All four are valid training examples of "cat"!
```

Important: augmentation is applied **only during training**, not during testing.

### Learning Rate Scheduling

Start with a larger learning rate for fast progress, then reduce it for
fine-tuning:

```
  Learning
  Rate
    ^
    |----\
    |     \
    |      ----\
    |           \
    |            ----\
    |                 \-----
    +-------------------------> Epoch
     Fast exploration   Fine-tuning
```

### Dropout

Randomly disable neurons during training to prevent overfitting:

```
  Without Dropout:              With Dropout (p=0.5):
  All neurons active            Random 50% disabled

  o---o---o---o                 o---x   o---o
  |   |   |   |                 |       |   |
  o---o---o---o                 o---o---x   o
  |   |   |   |                 |   |       |
  o---o---o---o                 x   o---o---o

  (x = disabled neuron, different each batch)
```

This forces the network to develop redundant representations and not
rely on any single neuron.

---

## 8. Running the Tutorials

### On Google Colab (Recommended for beginners)

1. Upload a notebook (`.ipynb`) to [Google Colab](https://colab.research.google.com/)
2. Set runtime to GPU: `Runtime > Change runtime type > GPU`
3. Run all cells: `Runtime > Run all`

### Locally with Python

```bash
# Install dependencies
pip install torch torchvision matplotlib

# Run each experiment
python 01_mnist_deep_learning.py
python 02_cifar10_deep_learning.py
python 03_cifar10_cnn.py

# Generate comparison report
python generate_report.py
```

Results are saved to `figures/` (plots) and JSON files (metrics).

---

## 9. Results Summary

| Experiment | Model | Parameters | Best Test Accuracy | Training Time |
| ---------- | ----- | ---------- | ------------------ | ------------- |
| MNIST FC | FC(784->256->128->10) | 235K | 98.1% | 50s |
| CIFAR-10 FC | FC(3072->512->256->128->10) | 1,739K | 55.0% | 138s |
| CIFAR-10 CNN | 6-conv + FC head | 815K | 86.9% | 329s |

See `report.md` for the full report with plots and per-class breakdowns.
