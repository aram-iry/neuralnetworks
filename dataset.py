import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    TRAIN_DIR, TRAIN_LABELS_CSV, LABEL_OFFSET, IMG_SIZE, 
    IMG_MEAN, IMG_STD, BATCH_SIZE, NUM_WORKERS, VAL_SPLIT, SEED
)

class FoodDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = str(self.data_frame.iloc[idx, 0])
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += '.jpg'
            
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Shift label from 1-80 to 0-79
        label = int(self.data_frame.iloc[idx, 1]) - LABEL_OFFSET
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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

def get_train_val_loaders():
    train_transform, val_transform = get_transforms()
    
    df = pd.read_csv(TRAIN_LABELS_CSV)
    labels = df.iloc[:, 1].values
    
    full_train_dataset = FoodDataset(TRAIN_LABELS_CSV, TRAIN_DIR, transform=train_transform)
    full_val_dataset = FoodDataset(TRAIN_LABELS_CSV, TRAIN_DIR, transform=val_transform)
    
    # Safely handle 100/0 split without crashing scikit-learn
    if VAL_SPLIT == 0.0:
        print("[Dataset] Using 100% of data for training. Validation is disabled.")
        train_idx = list(range(len(labels)))
        val_idx = []
    else:
        print(f"[Dataset] Splitting data {100*(1-VAL_SPLIT):.0f}/{100*VAL_SPLIT:.0f}")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    
    train_sub = Subset(full_train_dataset, train_idx)
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    
    if len(val_idx) > 0:
        val_sub = Subset(full_val_dataset, val_idx)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True)
    else:
        val_loader = [] 
        
    return train_loader, val_loader
