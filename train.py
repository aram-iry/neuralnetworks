"""
Training loop – GPU-accelerated with AMP and Mixup for scratch training.
"""

import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from config import (
    EPOCHS, EARLY_STOP_PATIENCE, CHECKPOINT_PATH, OUTPUT_DIR,
    BACKBONE_LR, HEAD_LR, WEIGHT_DECAY, SCHEDULER,
    STEP_SIZE, STEP_GAMMA, FREEZE_BACKBONE_EPOCHS, SEED,
)
from seed import seed_everything
from dataset import get_train_val_loaders
from model import build_model, get_optimizer


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    # Mixup parameter
    alpha = 1

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # --- MIXUP IMPLEMENTATION ---
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
            
        index = torch.randperm(imgs.size(0)).to(device)
        mixed_imgs = lam * imgs + (1 - lam) * imgs[index]
        labels_a, labels_b = labels, labels[index]
        # ----------------------------

        # New PyTorch AMP syntax (removes FutureWarnings)
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(mixed_imgs)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        
        # Track dominant label for training F1 calculation
        dominant_labels = labels_a if lam > 0.5 else labels_b
        all_labels.append(dominant_labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx + 1}/{len(loader)}", end="\r")

    n_samples = sum(len(p) for p in all_preds)
    epoch_loss = running_loss / n_samples
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(labels, preds, average="weighted")
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    n_samples = sum(len(p) for p in all_preds)
    epoch_loss = running_loss / n_samples
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(labels, preds, average="weighted")
    return epoch_loss, epoch_f1


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
    else:
        torch.set_num_threads(os.cpu_count())
        print(f"[INFO] Threads available: {torch.get_num_threads()}")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = get_train_val_loaders()
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────

    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {total_params:,} total, {trainable:,} trainable")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY)

    # New PyTorch AMP syntax (removes FutureWarnings)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    # ── Scheduler ─────────────────────────────────────────────────
    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA
        )

    # ── Training loop ─────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        FREEZE_BACKBONE_EPOCHS = 0
        # Unfreeze backbone (since FREEZE_BACKBONE_EPOCHS is 0, this triggers on Epoch 1)
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            print(f"\n[INFO] Epoch {epoch}: unfreezing backbone for full fine-tuning")
            
            optimizer = get_optimizer(model, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
            if SCHEDULER == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - epoch + 1, eta_min=1e-6
                )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA
                )

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f} | "
            f"{elapsed:.0f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Checkpoint best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    print(f"\n[DONE] Best validation weighted-F1: {best_f1:.4f}")
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    main()
