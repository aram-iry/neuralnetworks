"""
Dataset & DataLoader factories.
- Flat image directory + CSV labels for training
- Flat image directory for test
"""

import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torchvision.transforms as T

from config import (
    TRAIN_DIR, TEST_DIR, TRAIN_LABELS_CSV, LABEL_OFFSET,
    IMG_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, NUM_WORKERS, VAL_SPLIT, SEED,
)
from seed import seed_everything


# ──────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────
def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.08),
        T.RandomRotation(20),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(int(IMG_SIZE * 1.14)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


def get_test_transforms() -> T.Compose:
    return get_val_transforms()


def get_tta_transforms() -> Tuple[List[T.Compose], T.Compose]:
    """7 views: original, hflip, vflip, 4 corner crops."""
    resize_size = int(IMG_SIZE * 1.14)
    base = get_val_transforms()
    hflip = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])
    vflip = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(IMG_SIZE),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])
    # 4 corner crops using FiveCrop (top-left, top-right, bottom-left, bottom-right, center)
    five_crop = T.Compose([
        T.Resize(resize_size),
        T.FiveCrop(IMG_SIZE),
        T.Lambda(lambda crops: torch.stack([
            T.Normalize(IMG_MEAN, IMG_STD)(T.ToTensor()(crop)) for crop in crops
        ])),
    ])
    return [base, hflip, vflip], five_crop


# ──────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────
class FoodTrainDataset(Dataset):
    """Flat folder + CSV labels. Internal labels are 0-based."""

    def __init__(self, img_dir: str, csv_path: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        # Keep only rows whose image files actually exist
        self.filenames = []
        self.labels = []
        for _, row in df.iterrows():
            fpath = os.path.join(img_dir, row["img_name"])
            if os.path.exists(fpath):
                self.filenames.append(row["img_name"])
                self.labels.append(int(row["label"]) - LABEL_OFFSET)  # 0-based

        print(f"  [Dataset] Loaded {len(self.filenames)} images from {img_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class FoodTestDataset(Dataset):
    """Flat folder of unlabeled images."""

    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.fnames = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        print(f"  [Dataset] Found {len(self.fnames)} test images in {img_dir}")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


# ──────────────────────────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────────────────────────
def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def get_train_val_loaders():
    """Returns (train_loader, val_loader) with stratified split."""
    seed_everything()

    # Load full dataset twice (different transforms)
    full_ds = FoodTrainDataset(TRAIN_DIR, TRAIN_LABELS_CSV, transform=None)
    labels = np.array(full_ds.labels)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SPLIT, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    train_ds = FoodTrainDataset(TRAIN_DIR, TRAIN_LABELS_CSV, transform=get_train_transforms())
    val_ds = FoodTrainDataset(TRAIN_DIR, TRAIN_LABELS_CSV, transform=get_val_transforms())

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader


def get_test_loader():
    ds = FoodTestDataset(TEST_DIR, transform=get_test_transforms())
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
