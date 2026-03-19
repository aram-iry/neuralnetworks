import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CHECKPOINT_PATH, EARLY_STOP_PATIENCE
)
from model import build_model
from dataset import get_train_val_loaders

# --- MIXUP / CUTMIX HELPERS ---
def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- VALIDATION HELPER ---
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = get_train_val_loaders()
    
    model = build_model(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    best_f1 = 0.0
    patience_counter = 0

    print("[INFO] Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_labels = [], []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            all_train_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_train_labels.append(labels_a.cpu().numpy()) # Track primary label for F1
            
        scheduler.step()
        
        n_train_samples = sum(len(p) for p in all_train_preds)
        train_loss = train_loss / n_train_samples
        train_preds = np.concatenate(all_train_preds)
        train_labels = np.concatenate(all_train_labels)
        train_f1 = f1_score(train_labels, train_preds, average="weighted")
        
        epoch_time = time.time() - start_time
        
        # --- VALIDATION / SAVE ---
        if len(val_loader) > 0:
            val_loss, val_f1 = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | {epoch_time:.0f}s")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  ✓ Saved best model (Val F1={best_f1:.4f})")
            else:
                patience_counter += 1
        else:
            # 100/0 mode: Print train only, save on best Train F1
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | {epoch_time:.0f}s")
            
            if train_f1 > best_f1:
                best_f1 = train_f1
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  ✓ Saved best model (Train F1={best_f1:.4f})")
            else:
                patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

if __name__ == "__main__":
    main()
