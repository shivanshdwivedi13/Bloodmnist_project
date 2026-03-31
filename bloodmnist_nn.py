"""
==============================================
  BloodMNIST Classification - Simple Neural Network (PyTorch)
==============================================
BloodMNIST: 28x28 RGB microscopic blood cell images
8 classes: basophil, eosinophil, erythroblast, ig, lymphocyte,
           monocyte, neutrophil, platelet
==============================================
"""

# ──────────────────────────────────────────
# 1. INSTALL & IMPORTS
# ──────────────────────────────────────────
# Run this once in your terminal:
#   pip install medmnist torch torchvision matplotlib scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import medmnist
from medmnist import BloodMNIST

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ──────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 0.001
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Basophil", "Eosinophil", "Erythroblast", "IG",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
]

print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────
# 3. DATA LOADING
# ──────────────────────────────────────────
# Normalize to ImageNet mean/std (works well for RGB medical images)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = BloodMNIST(split="train", transform=transform, download=True)
val_dataset   = BloodMNIST(split="val",   transform=transform, download=True)
test_dataset  = BloodMNIST(split="test",  transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ──────────────────────────────────────────
# 4. VISUALIZE SAMPLE IMAGES
# ──────────────────────────────────────────
def show_samples(dataset, n=8):
    fig, axes = plt.subplots(1, n, figsize=(16, 2.5))
    indices = np.random.randint(0, len(dataset), n)
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        # Un-normalize for display
        img = img * 0.5 + 0.5
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(CLASS_NAMES[label[0]], fontsize=8)
        axes[i].axis("off")
    plt.suptitle("BloodMNIST Sample Images", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()
    print("Saved: sample_images.png")

show_samples(train_dataset)

# ──────────────────────────────────────────
# 5A. MODEL 1 — Simple Fully Connected Network (MLP)
# ──────────────────────────────────────────
class SimpleMLP(nn.Module):
    """
    Flatten the 28x28x3 image → feed through dense layers.
    Input:  3 x 28 x 28 = 2352 features
    Output: 8 class logits
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                      # 2352
            nn.Linear(3 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)                  # 8 classes
        )

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────
# 5B. MODEL 2 — Simple CNN (better for images)
# ──────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    2 Conv layers → Global Average Pooling → Fully Connected
    Much better at capturing spatial features in images.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 28x28x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 14x14x32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 7x7x64

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 7x7x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 1x1x128 (Global Avg Pool)
            nn.Flatten(),              # 128
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 8)           # 8 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────
# 6. TRAINING & EVALUATION FUNCTIONS
# ──────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*50}")
    print(f"  Training: {model_name}")
    print(f"{'='*50}")

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")
    return model, history


# ──────────────────────────────────────────
# 7. TRAIN BOTH MODELS
# ──────────────────────────────────────────
mlp_model, mlp_history = train_model(SimpleMLP(), "MLP")
cnn_model, cnn_history = train_model(SimpleCNN(), "CNN")


# ──────────────────────────────────────────
# 8. PLOT TRAINING CURVES
# ──────────────────────────────────────────
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title(f"{title} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title(f"{title} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"{title.replace(' ', '_')}_curves.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

plot_history(mlp_history, "MLP Model")
plot_history(cnn_history, "CNN Model")


# ──────────────────────────────────────────
# 9. TEST SET EVALUATION + CONFUSION MATRIX
# ──────────────────────────────────────────
def full_evaluation(model, model_name):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n[{model_name}] Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    # Collect all predictions
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().numpy())

    # Classification Report
    print(f"\nClassification Report — {model_name}:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = f"{model_name}_confusion_matrix.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


full_evaluation(mlp_model, "MLP")
full_evaluation(cnn_model, "CNN")


# ──────────────────────────────────────────
# 10. COMPARE MODELS SUMMARY
# ──────────────────────────────────────────
print("\n" + "="*50)
print("  MODEL COMPARISON SUMMARY")
print("="*50)
print(f"{'Model':<10} {'Final Val Acc':>15} {'Final Val Loss':>15}")
print("-"*42)
print(f"{'MLP':<10} {mlp_history['val_acc'][-1]*100:>14.2f}% {mlp_history['val_loss'][-1]:>15.4f}")
print(f"{'CNN':<10} {cnn_history['val_acc'][-1]*100:>14.2f}% {cnn_history['val_loss'][-1]:>15.4f}")
print("="*50)
print("\nDone! CNN typically outperforms MLP on image data.")
print("Saved model weights: best_MLP.pth, best_CNN.pth")
