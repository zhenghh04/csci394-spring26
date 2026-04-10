#!/usr/bin/env python3
"""CIFAR-10 Classification with a Convolutional Neural Network.

Trains a 6-conv-layer CNN (3 blocks of Conv-BN-ReLU-Conv-BN-ReLU-MaxPool)
with data augmentation, batch normalization, dropout, and LR scheduling.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Configuration ──────────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 1e-3
SEED = 42
DATA_DIR = "./data"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

torch.manual_seed(SEED)

# ── Device ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Data (with augmentation for training) ──────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}  Test samples: {len(test_dataset)}")

# ── Model ──────────────────────────────────────────────────────────────
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CIFAR10CNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# ── Helpers ────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss = total_correct = total_seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen

# ── Training ───────────────────────────────────────────────────────────
history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "lr": []}

print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'LR':>10} | {'Time':>6}")
print("-" * 78)

t0 = time.time()
best_test_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    ep_start = time.time()
    model.train()
    run_loss = run_correct = run_seen = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * labels.size(0)
        run_correct += (logits.argmax(1) == labels).sum().item()
        run_seen += labels.size(0)

    tr_loss = run_loss / run_seen
    tr_acc = run_correct / run_seen
    te_loss, te_acc = evaluate(model, test_loader)
    scheduler.step(te_loss)
    lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(tr_loss)
    history["test_loss"].append(te_loss)
    history["train_acc"].append(tr_acc)
    history["test_acc"].append(te_acc)
    history["lr"].append(lr)

    marker = " *" if te_acc > best_test_acc else ""
    best_test_acc = max(best_test_acc, te_acc)
    print(f"{epoch:5d} | {tr_loss:10.4f} | {tr_acc:8.2%} | {te_loss:9.4f} | {te_acc:7.2%} | {lr:10.6f} | {time.time()-ep_start:5.1f}s{marker}")

total_time = time.time() - t0
print(f"\nTotal training time: {total_time:.1f}s")
print(f"Best test accuracy:  {best_test_acc:.2%}")

# ── Predictions & confusion matrix ────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

confusion = torch.zeros(10, 10, dtype=torch.int64)
for p, t in zip(all_preds, all_labels):
    confusion[t, p] += 1

per_class_acc = {}
for i in range(10):
    total = confusion[i].sum().item()
    correct = confusion[i, i].item()
    per_class_acc[CLASS_NAMES[i]] = round(correct / total, 4)

# ── Save figures ───────────────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train", ms=3)
axes[0].plot(epochs_range, history["test_loss"], "r-o", label="Test", ms=3)
axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, [a*100 for a in history["train_acc"]], "b-o", label="Train", ms=3)
axes[1].plot(epochs_range, [a*100 for a in history["test_acc"]], "r-o", label="Test", ms=3)
axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history["lr"], "g-o", ms=3)
axes[2].set(xlabel="Epoch", ylabel="Learning Rate", title="LR Schedule")
axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

plt.suptitle("CIFAR-10 — CNN", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "03_cifar10_cnn_curves.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(confusion, cmap="Blues")
for i in range(10):
    for j in range(10):
        c = "white" if confusion[i, j] > confusion.max() / 2 else "black"
        ax.text(j, i, f"{confusion[i,j]}", ha="center", va="center", color=c, fontsize=8)
ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
plt.colorbar(im); plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "03_cifar10_cnn_confusion.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
accs = list(per_class_acc.values())
colors = ["green" if a > 0.8 else "orange" if a > 0.6 else "red" for a in accs]
ax.bar(range(10), [a*100 for a in accs], color=colors, edgecolor="black")
ax.set_xticks(range(10))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set(ylabel="Accuracy (%)", title="Per-Class Test Accuracy (CNN)")
ax.axhline(np.mean(accs)*100, color="blue", ls="--", label=f"Mean: {np.mean(accs):.1%}")
ax.legend(); plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "03_cifar10_cnn_perclass.png"), dpi=150)
plt.close()

# ── Save JSON report ──────────────────────────────────────────────────
report = {
    "experiment": "CIFAR-10 — CNN",
    "model": "CNN: 3x[Conv-BN-ReLU-Conv-BN-ReLU-MaxPool] + FC(256->10)",
    "parameters": total_params,
    "device": str(device),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "data_augmentation": True,
    "lr_scheduling": "ReduceLROnPlateau(factor=0.5, patience=3)",
    "final_train_loss": round(history["train_loss"][-1], 4),
    "final_test_loss": round(history["test_loss"][-1], 4),
    "final_train_acc": round(history["train_acc"][-1], 4),
    "final_test_acc": round(history["test_acc"][-1], 4),
    "best_test_acc": round(best_test_acc, 4),
    "training_time_s": round(total_time, 2),
    "per_class_acc": per_class_acc,
    "history": history,
}
with open(os.path.join(OUT_DIR, "03_cifar10_cnn_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\nFinal test accuracy: {history['test_acc'][-1]:.2%}")
print(f"Saved: 03_cifar10_cnn_curves.png, 03_cifar10_cnn_confusion.png, 03_cifar10_cnn_perclass.png, 03_cifar10_cnn_report.json")
