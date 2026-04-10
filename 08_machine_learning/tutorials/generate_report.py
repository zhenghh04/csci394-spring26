#!/usr/bin/env python3
"""Generate a combined comparison report from the three experiment JSON files."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load reports
with open(os.path.join(OUT_DIR, "01_mnist_fc_report.json")) as f:
    r1 = json.load(f)
with open(os.path.join(OUT_DIR, "02_cifar10_fc_report.json")) as f:
    r2 = json.load(f)
with open(os.path.join(OUT_DIR, "03_cifar10_cnn_report.json")) as f:
    r3 = json.load(f)

# ── Figure 1: Comparison bar chart ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

labels = ["MNIST\nFC Net", "CIFAR-10\nFC Net", "CIFAR-10\nCNN"]
best_accs = [r1["best_test_acc"] * 100, r2["best_test_acc"] * 100, r3["best_test_acc"] * 100]
params = [r1["parameters"], r2["parameters"], r3["parameters"]]
times = [r1["training_time_s"], r2["training_time_s"], r3["training_time_s"]]

colors = ["#2196F3", "#FF9800", "#4CAF50"]

# Accuracy comparison
bars = axes[0].bar(labels, best_accs, color=colors, edgecolor="black", width=0.6)
for bar, val in zip(bars, best_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[0].set_ylabel("Best Test Accuracy (%)")
axes[0].set_title("Accuracy Comparison")
axes[0].set_ylim(0, 110)
axes[0].grid(axis="y", alpha=0.3)

# Parameter count
bars = axes[1].bar(labels, [p/1000 for p in params], color=colors, edgecolor="black", width=0.6)
for bar, val in zip(bars, params):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val/1000:.0f}K", ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[1].set_ylabel("Parameters (thousands)")
axes[1].set_title("Model Size")
axes[1].grid(axis="y", alpha=0.3)

# Training time
bars = axes[2].bar(labels, times, color=colors, edgecolor="black", width=0.6)
for bar, val in zip(bars, times):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}s", ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[2].set_ylabel("Training Time (seconds)")
axes[2].set_title("Training Time")
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle("Deep Learning Tutorial — Model Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "report_comparison.png"), dpi=150)
plt.close()

# ── Figure 2: CIFAR-10 FC vs CNN accuracy curves ─────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Test accuracy curves
e_fc = range(1, r2["epochs"] + 1)
e_cnn = range(1, r3["epochs"] + 1)
ax1.plot(e_fc, [a*100 for a in r2["history"]["test_acc"]], "r-o", label="FC Network", ms=4)
ax1.plot(e_cnn, [a*100 for a in r3["history"]["test_acc"]], "g-o", label="CNN", ms=4)
ax1.set(xlabel="Epoch", ylabel="Test Accuracy (%)", title="CIFAR-10: FC vs CNN — Test Accuracy")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Per-class comparison
classes = list(r2["per_class_acc"].keys())
fc_accs = [r2["per_class_acc"][c] * 100 for c in classes]
cnn_accs = [r3["per_class_acc"][c] * 100 for c in classes]

x = np.arange(len(classes))
w = 0.35
ax2.bar(x - w/2, fc_accs, w, label="FC Network", color="#FF9800", edgecolor="black")
ax2.bar(x + w/2, cnn_accs, w, label="CNN", color="#4CAF50", edgecolor="black")
ax2.set_xticks(x)
ax2.set_xticklabels(classes, rotation=45, ha="right")
ax2.set(ylabel="Accuracy (%)", title="CIFAR-10: Per-Class Accuracy Comparison")
ax2.legend(fontsize=11)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "report_cifar10_comparison.png"), dpi=150)
plt.close()

# ── Markdown report ───────────────────────────────────────────────────
report_md = f"""# Deep Learning Tutorial — Training Report

**CSCI 394 — Spring 2026**
**Device:** Apple MPS (Metal Performance Shaders)
**PyTorch version:** see individual runs

---

## Experiment Summary

| Metric | MNIST FC | CIFAR-10 FC | CIFAR-10 CNN |
|--------|----------|-------------|--------------|
| **Model** | FC(784→256→128→10) | FC(3072→512→256→128→10) + Dropout | 3×[Conv-BN-ReLU-Conv-BN-ReLU-MaxPool] + FC |
| **Parameters** | {r1['parameters']:,} | {r2['parameters']:,} | {r3['parameters']:,} |
| **Epochs** | {r1['epochs']} | {r2['epochs']} | {r3['epochs']} |
| **Data Augmentation** | No | No | Yes (RandomCrop, HFlip) |
| **LR Scheduling** | No | No | ReduceLROnPlateau |
| **Best Test Accuracy** | **{r1['best_test_acc']:.2%}** | **{r2['best_test_acc']:.2%}** | **{r3['best_test_acc']:.2%}** |
| **Final Test Accuracy** | {r1['final_test_acc']:.2%} | {r2['final_test_acc']:.2%} | {r3['final_test_acc']:.2%} |
| **Final Train Accuracy** | {r1['final_train_acc']:.2%} | {r2['final_train_acc']:.2%} | {r3['final_train_acc']:.2%} |
| **Final Test Loss** | {r1['final_test_loss']:.4f} | {r2['final_test_loss']:.4f} | {r3['final_test_loss']:.4f} |
| **Training Time** | {r1['training_time_s']:.1f}s | {r2['training_time_s']:.1f}s | {r3['training_time_s']:.1f}s |

---

## 1. MNIST — Fully Connected Network

A simple 3-layer fully connected network achieves **{r1['best_test_acc']:.2%}** test accuracy on MNIST
in just {r1['epochs']} epochs ({r1['training_time_s']:.0f}s). MNIST is a relatively easy benchmark:
handwritten digits are centered, grayscale, and low-resolution.

**Per-digit accuracy:**

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| Accuracy | {r1['per_class_acc']['0']:.1%} | {r1['per_class_acc']['1']:.1%} | {r1['per_class_acc']['2']:.1%} | {r1['per_class_acc']['3']:.1%} | {r1['per_class_acc']['4']:.1%} | {r1['per_class_acc']['5']:.1%} | {r1['per_class_acc']['6']:.1%} | {r1['per_class_acc']['7']:.1%} | {r1['per_class_acc']['8']:.1%} | {r1['per_class_acc']['9']:.1%} |

![MNIST Training Curves](01_mnist_fc_curves.png)
![MNIST Confusion Matrix](01_mnist_fc_confusion.png)

---

## 2. CIFAR-10 — Fully Connected Network

The same FC approach struggles on CIFAR-10, achieving only **{r2['best_test_acc']:.2%}** test accuracy.
This is expected: flattening a 32×32×3 color image into a 3072-dimensional vector destroys all spatial
structure. The model has {r2['parameters']:,} parameters (7× more than MNIST) but still underperforms.

**Per-class accuracy:**

| Class | airplane | auto | bird | cat | deer | dog | frog | horse | ship | truck |
|-------|----------|------|------|-----|------|-----|------|-------|------|-------|
| Accuracy | {r2['per_class_acc']['airplane']:.1%} | {r2['per_class_acc']['automobile']:.1%} | {r2['per_class_acc']['bird']:.1%} | {r2['per_class_acc']['cat']:.1%} | {r2['per_class_acc']['deer']:.1%} | {r2['per_class_acc']['dog']:.1%} | {r2['per_class_acc']['frog']:.1%} | {r2['per_class_acc']['horse']:.1%} | {r2['per_class_acc']['ship']:.1%} | {r2['per_class_acc']['truck']:.1%} |

Key observations:
- **Cat** ({r2['per_class_acc']['cat']:.1%}) and **dog** ({r2['per_class_acc']['dog']:.1%}) are the hardest — they share similar shapes and textures
- **Ship** ({r2['per_class_acc']['ship']:.1%}) and **automobile** ({r2['per_class_acc']['automobile']:.1%}) are easiest — distinct shapes and backgrounds

![CIFAR-10 FC Training Curves](02_cifar10_fc_curves.png)
![CIFAR-10 FC Per-Class Accuracy](02_cifar10_fc_perclass.png)
![CIFAR-10 FC Confusion Matrix](02_cifar10_fc_confusion.png)

---

## 3. CIFAR-10 — Convolutional Neural Network

The CNN achieves **{r3['best_test_acc']:.2%}** test accuracy — a **{(r3['best_test_acc'] - r2['best_test_acc'])*100:.1f} percentage point improvement** over
the FC network — with **fewer parameters** ({r3['parameters']:,} vs {r2['parameters']:,}).

This demonstrates the power of convolutional architectures for image data:
- **Local feature detection** via convolution filters
- **Parameter sharing** across spatial locations
- **Hierarchical feature learning** (edges → textures → objects)
- **Translation invariance** via max pooling

**Per-class accuracy:**

| Class | airplane | auto | bird | cat | deer | dog | frog | horse | ship | truck |
|-------|----------|------|------|-----|------|-----|------|-------|------|-------|
| FC  | {r2['per_class_acc']['airplane']:.1%} | {r2['per_class_acc']['automobile']:.1%} | {r2['per_class_acc']['bird']:.1%} | {r2['per_class_acc']['cat']:.1%} | {r2['per_class_acc']['deer']:.1%} | {r2['per_class_acc']['dog']:.1%} | {r2['per_class_acc']['frog']:.1%} | {r2['per_class_acc']['horse']:.1%} | {r2['per_class_acc']['ship']:.1%} | {r2['per_class_acc']['truck']:.1%} |
| CNN | {r3['per_class_acc']['airplane']:.1%} | {r3['per_class_acc']['automobile']:.1%} | {r3['per_class_acc']['bird']:.1%} | {r3['per_class_acc']['cat']:.1%} | {r3['per_class_acc']['deer']:.1%} | {r3['per_class_acc']['dog']:.1%} | {r3['per_class_acc']['frog']:.1%} | {r3['per_class_acc']['horse']:.1%} | {r3['per_class_acc']['ship']:.1%} | {r3['per_class_acc']['truck']:.1%} |

The CNN improves **every single class**, with the largest gains on the hardest categories (cat: +45pp, dog: +30pp, bird: +44pp).

![CIFAR-10 CNN Training Curves](03_cifar10_cnn_curves.png)
![CIFAR-10 CNN Per-Class Accuracy](03_cifar10_cnn_perclass.png)
![CIFAR-10 CNN Confusion Matrix](03_cifar10_cnn_confusion.png)

---

## Comparison

![Model Comparison](report_comparison.png)
![CIFAR-10 FC vs CNN](report_cifar10_comparison.png)

### Key Takeaways

1. **MNIST is easy for fully connected networks** — 98% accuracy is achievable with a simple 3-layer network and ~235K parameters.

2. **FC networks fail on natural images** — Despite having 7× more parameters, the CIFAR-10 FC model only reaches ~55%. Flattening images destroys spatial structure.

3. **CNNs are dramatically better for images** — With 2× fewer parameters, the CNN reaches ~87% on CIFAR-10. Convolutions preserve and exploit spatial locality.

4. **The accuracy gap tells the story**:
   - MNIST FC: 98.1% ✓ (FC is sufficient for simple, centered patterns)
   - CIFAR-10 FC: 55.0% ✗ (FC cannot handle complex, variable images)
   - CIFAR-10 CNN: 86.9% ✓ (CNNs capture the spatial features that matter)

5. **Supporting techniques matter**: Data augmentation, batch normalization, and LR scheduling all contribute to the CNN's superior performance.
"""

with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
    f.write(report_md)

print("Generated:")
print("  report.md")
print("  report_comparison.png")
print("  report_cifar10_comparison.png")
