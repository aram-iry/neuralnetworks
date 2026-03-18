import os
import time
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.model_selection import KFold

# Import your existing configurations and model
from config import (
    TRAIN_DIR, TRAIN_LABELS_CSV, LABEL_OFFSET, 
    IMG_SIZE, IMG_MEAN, IMG_STD, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY
)
from model import build_model

# ─── Custom CSV Dataset ──────────────────────────────────────────
class FoodCSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Assuming column 0 is image name, column 1 is label
        img_name = str(self.data_frame.iloc[idx, 0])
        
        # Some CSVs don't include the .jpg extension, add it if missing
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += '.jpg'
            
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # PyTorch expects labels to start at 0. We subtract your LABEL_OFFSET (1)
        label = int(self.data_frame.iloc[idx, 1]) - LABEL_OFFSET
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ─── Transforms ──────────────────────────────────────────────────
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        v2.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])
    return train_transform, val_transform

# ─── Mixup Helpers ───────────────────────────────────────────────
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    # Use our custom CSV dataset instead of ImageFolder
    full_dataset_train = FoodCSVDataset(TRAIN_LABELS_CSV, TRAIN_DIR, transform=train_transform)
    full_dataset_val = FoodCSVDataset(TRAIN_LABELS_CSV, TRAIN_DIR, transform=val_transform)
    
    dataset_size = len(full_dataset_train)
    indices = list(range(dataset_size))
    
    # 5-Fold Setup
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    os.makedirs("outputs", exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n{'='*20} FOLD {fold} / {k_folds} {'='*20}")
        
        train_sub = Subset(full_dataset_train, train_idx)
        val_sub = Subset(full_dataset_val, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = build_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        best_acc = 0.0
        
        for epoch in range(1, EPOCHS + 1):
            start_time = time.time()
            
            # --- TRAIN ---
            model.train()
            train_loss, correct_train, total_train = 0.0, 0, 0
            
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
                _, predicted = outputs.max(1)
                correct_train += (lam * predicted.eq(labels_a).sum().float() + 
                                  (1 - lam) * predicted.eq(labels_b).sum().float()).item()
                total_train += labels.size(0)
                
            scheduler.step()
            train_loss, train_acc = train_loss / total_train, correct_train / total_train
            
            # --- VALIDATION ---
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()
            
            val_loss, val_acc = val_loss / total_val, correct_val / total_val
            
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | {time.time()-start_time:.0f}s")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"outputs/best_model_fold_{fold}.pth")
                print(f"  ✓ Saved Fold {fold} best model (Acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
