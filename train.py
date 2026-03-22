import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from config import *
from seed import seed_everything
from model_msnet import build_model
from dataset import get_train_val_loaders

# --- Reproducibility: must be called before any model/data initialization ---
seed_everything()

def top_k_accuracy(outputs, labels, k):
    """Compute top-k accuracy given raw logits and integer labels."""
    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().sum().item()

def compute_grad_norm(model):
    """Compute total L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return total_norm ** 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=IMG_SIZE,
                        help=f"Input image size (default: {IMG_SIZE})")
    args = parser.parse_args()

    img_size = args.img_size
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{img_size}.pth")
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{img_size}.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Initialise CSV log ---
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "elapsed_sec", "epoch_duration_sec", "samples_per_sec",
            "train_loss", "val_loss",
            "train_top1_acc", "val_top1_acc", "val_top5_acc",
            "train_micro_f1", "train_macro_f1",
            "val_micro_f1", "val_macro_f1",
            "grad_norm", "lr"
        ])

    train_loader, val_loader = get_train_val_loaders(img_size=img_size)
    n_train = len(train_loader.dataset)

    model = build_model().to(DEVICE)

    # --- Model size in MB (parameter memory, consistent across runs) ---
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n[INFO] IMG_SIZE: {img_size} | Seed: {SEED} | Device: {DEVICE}")
    print(f"[INFO] Model size: {model_size_mb:.2f} MB")
    print(f"[INFO] Metrics will be saved to: {metrics_path}")

    best_f1 = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        train_top1_correct = 0
        train_total = 0
        train_preds, train_labels_all = [], []
        grad_norm_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Grad norm before optimizer step
            grad_norm_accum += compute_grad_norm(model)

            optimizer.step()
            train_loss += loss.item()
            train_top1_correct += top_k_accuracy(outputs, labels, k=1)
            train_total += labels.size(0)
            train_preds.append(outputs.argmax(1).cpu().numpy())
            train_labels_all.append(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # --- TRAIN METRICS ---
        avg_grad_norm = grad_norm_accum / len(train_loader)
        train_top1_acc = train_top1_correct / train_total
        train_preds_np = np.concatenate(train_preds)
        train_targets_np = np.concatenate(train_labels_all)
        train_micro_f1 = f1_score(train_targets_np, train_preds_np, average='micro')
        train_macro_f1 = f1_score(train_targets_np, train_preds_np, average='macro')

        # --- VALIDATION PHASE ---
        model.eval()
        val_preds, val_labels_all = [], []
        val_top1_correct = 0
        val_top5_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)

                val_loss += criterion(outputs, labels).item()
                val_top1_correct += top_k_accuracy(outputs, labels, k=1)
                val_top5_correct += top_k_accuracy(outputs, labels, k=5)
                val_total += labels.size(0)

                val_preds.append(outputs.argmax(1).cpu().numpy())
                val_labels_all.append(labels.cpu().numpy())

        # --- VAL METRICS ---
        val_preds_np = np.concatenate(val_preds)
        val_targets_np = np.concatenate(val_labels_all)
        avg_val_loss = val_loss / len(val_loader)
        val_top1_acc = val_top1_correct / val_total
        val_top5_acc = val_top5_correct / val_total
        val_micro_f1 = f1_score(val_targets_np, val_preds_np, average='micro')
        val_macro_f1 = f1_score(val_targets_np, val_preds_np, average='macro')

        # --- TIMING ---
        epoch_duration = time.time() - epoch_start
        elapsed = time.time() - train_start
        samples_per_sec = n_train / epoch_duration
        current_lr = scheduler.get_last_lr()[0]

        print(f" >> Epoch {epoch} | "
              f"T-Loss: {train_loss/len(train_loader):.4f} | V-Loss: {avg_val_loss:.4f} | "
              f"T-Top1: {train_top1_acc*100:.2f}% | V-Top1: {val_top1_acc*100:.2f}% | V-Top5: {val_top5_acc*100:.2f}% | "
              f"T-MacroF1: {train_macro_f1:.4f} | V-MacroF1: {val_macro_f1:.4f} | "
              f"GradNorm: {avg_grad_norm:.4f} | {samples_per_sec:.1f} samp/s | LR: {current_lr:.6f}")

        # --- SAVE METRICS ROW ---
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, round(elapsed, 2), round(epoch_duration, 2), round(samples_per_sec, 2),
                round(train_loss / len(train_loader), 6), round(avg_val_loss, 6),
                round(train_top1_acc, 6), round(val_top1_acc, 6), round(val_top5_acc, 6),
                round(train_micro_f1, 6), round(train_macro_f1, 6),
                round(val_micro_f1, 6), round(val_macro_f1, 6),
                round(avg_grad_norm, 6), current_lr
            ])

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f" >> [SAVED] New best model (Val Macro-F1): {best_f1:.4f}\n")
        else:
            print(f" >> [STAGNANT] Best Val Macro-F1 remains: {best_f1:.4f}\n")

if __name__ == "__main__":
    main()
