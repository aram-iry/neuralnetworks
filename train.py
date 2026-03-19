import os
import time
import warnings
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from config import *
from model import build_model
from dataset import get_train_val_loaders

# Mute the DML fallback warnings
warnings.filterwarnings("ignore")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_loader, val_loader = get_train_val_loaders()
    
    model = build_model().to(DEVICE)
    
    # We start with a slightly lower LR to be more precise
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) 
    
    # ReduceLROnPlateau will "rescue" the model if F1 stops climbing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"\n[INFO] Starting training on {DEVICE}")
    best_f1 = 0

    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # --- VALIDATION PHASE ---
        model.eval()
        all_preds, all_labels = [], []
        # Initialize val_f1 outside the loop so it's accessible to the scheduler
        current_val_f1 = 0.0 
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                all_preds.append(outputs.argmax(1).cpu().numpy())
                all_labels.append(labels.numpy())
        
        # Calculate F1 after the loop finishes
        if len(all_preds) > 0:
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_labels)
            current_val_f1 = f1_score(targets, preds, average='weighted')
        
        # --- LOGGING & SCHEDULING ---
        avg_train_loss = train_loss / len(train_loader)
        print(f" >> Results: Loss {avg_train_loss:.4f} | Val F1: {current_val_f1:.4f}")
        
        # Step the scheduler based on the F1 we just calculated
        scheduler.step(current_val_f1)
        
        if current_val_f1 > best_f1:
            best_f1 = current_val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f" >> [SAVED] New best model: {best_f1:.4f}\n")
        else:
            print(f" >> [STAGNANT] Best F1 remains: {best_f1:.4f}\n")

if __name__ == "__main__":
    main()