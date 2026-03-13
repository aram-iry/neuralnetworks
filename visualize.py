"""
Post-training visualizations:
  - Loss & F1 curves
  - Confusion matrix on the validation set
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from config import (
    OUTPUT_DIR, CHECKPOINT_PATH, NUM_CLASSES, CLASS_LIST_FILE, SEED,
)
from seed import seed_everything
from dataset import get_train_val_loaders
from model import build_model


def load_class_names():
    """Parse class_list_food.txt → list of names (index 0 = class 1)."""
    names = [""] * NUM_CLASSES
    with open(CLASS_LIST_FILE) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx = int(parts[0]) - 1  # 0-based
                names[idx] = parts[1]
    return names


def plot_curves():
    history_path = os.path.join(OUTPUT_DIR, "history.json")
    if not os.path.exists(history_path):
        print("[WARN] history.json not found – skipping curves")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, h["train_loss"], label="Train")
    axes[0].plot(epochs, h["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, h["train_f1"], label="Train")
    axes[1].plot(epochs, h["val_f1"], label="Val")
    axes[1].set_title("Weighted F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[DONE] Curves saved to {out}")


@torch.no_grad()
def plot_confusion_matrix():
    seed_everything(SEED)
    device = torch.device("cpu")
    class_names = load_class_names()

    _, val_loader = get_train_val_loaders()
    model = build_model(pretrained=False).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, annot=False, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Validation)")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[DONE] Confusion matrix saved to {out}")

    val_f1 = f1_score(labels, preds, average="weighted")
    print(f"\nValidation Weighted F1: {val_f1:.4f}")

    report = classification_report(labels, preds, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Validation Weighted F1: {val_f1:.4f}\n\n")
        f.write(report)


if __name__ == "__main__":
    plot_curves()
    plot_confusion_matrix()
