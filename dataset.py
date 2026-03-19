import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as T
from config import *

class FoodDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        if csv_file:
            df = pd.read_csv(csv_file)
            self.samples = df["img_name"].tolist()
            self.labels = [int(l) - LABEL_OFFSET for l in df["label"].tolist()]
        else:
            self.samples = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.labels = None

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.samples[idx])).convert("RGB")
        if self.transform: img = self.transform(img)
        return (img, self.labels[idx]) if self.labels is not None else (img, self.samples[idx])

def get_train_val_loaders():
    tfm = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_ds = FoodDataset(TRAIN_DIR, TRAIN_LABELS_CSV, transform=tfm)
    labels = np.array(full_ds.labels)
    split = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    train_idx, val_idx = next(split.split(np.zeros(len(labels)), labels))
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader