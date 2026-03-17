"""
Training loop - GPU-accelerated with AMP and gradient accumulation.
Auto-detects CUDA; falls back to CPU when unavailable.
"""

import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

from config import (
    DEVICE, EPOCHS, EARLY_STOP_PATIENCE, CHECKPOINT_PATH, OUTPUT_DIR,
    BACKBONE_LR, HEAD_LR, WEIGHT_DECAY, SCHEDULER,
    STEP_SIZE, STEP_GAMMA, FREEZE_BACKBONE_EPOCHS, SEED,
    USE_AMP, GRAD_ACCUM_STEPS,
)
from seed import seed_everything
from dataset import get_train_val_loaders
from model import build_model, unfreeze_backbone, get_optimizer


def _log_vram():
    """Print current and peak VRAM usage if CUDA is available."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024 ** 3
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[VRAM] current={used:.2f} GB  peak={peak:.2f} GB")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, grad_accum_steps):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        # Scale loss for gradient accumulation; track the unscaled value
        running_loss += loss.item() * imgs.size(0)
        scaler.scale(loss / grad_accum_steps).backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        all_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx + 1}/{len(loader)}", end="\r")

    n_samples = sum(len(p) for p in all_preds)
    epoch_loss = running_loss / n_samples
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    epoch_f1 = f1_score(labels_arr, preds, average="weighted")
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    n_samples = sum(len(p) for p in all_preds)
    epoch_loss = running_loss / n_samples
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    epoch_f1 = f1_score(labels_arr, preds, average="weighted")
    return epoch_loss, epoch_f1


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = DEVICE
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"[INFO] AMP enabled: {USE_AMP}")
        torch.backends.cudnn.benchmark = True  # speed up fixed-size input training
    else:
        print(f"[INFO] Threads available: {torch.get_num_threads()}")
        torch.set_num_threads(os.cpu_count())

    # AMP scaler (no-op on CPU)
    scaler = GradScaler(enabled=USE_AMP)

    # -- Data ------------------------------------------------------------
    train_loader, val_loader = get_train_val_loaders()
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"[INFO] Effective batch size: {train_loader.batch_size * GRAD_ACCUM_STEPS}")

    # -- Model -----------------------------------------------------------
    model = build_model(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {total_params:,} total, {trainable:,} trainable")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY)

    # -- Scheduler -------------------------------------------------------
    def _make_scheduler(opt, remaining_epochs):
        if SCHEDULER == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=remaining_epochs, eta_min=1e-6
            )
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=STEP_SIZE, gamma=STEP_GAMMA
        )

    scheduler = _make_scheduler(optimizer, EPOCHS)

    # -- Training loop ---------------------------------------------------
    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Unfreeze backbone after N warm-up epochs
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            print(f"\n[INFO] Epoch {epoch}: unfreezing backbone for full fine-tuning")
            unfreeze_backbone(model)
            optimizer = get_optimizer(model, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY)
            scheduler = _make_scheduler(optimizer, EPOCHS - epoch + 1)

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, USE_AMP, GRAD_ACCUM_STEPS,
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device, USE_AMP)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f} | "
            f"{elapsed:.0f}s"
        )
        if device.type == "cuda":
            _log_vram()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Checkpoint best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  checkpoint saved (F1={best_f1:.4f})")
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
