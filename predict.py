"""
Inference with optional TTA → submission.csv
Labels are converted back to 1-based for Kaggle submission.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_PATH, SUBMISSION_PATH, OUTPUT_DIR,
    TEST_DIR, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES,
    LABEL_OFFSET, SEED,
)
from seed import seed_everything
from dataset import FoodTestDataset, get_test_transforms, get_tta_transforms
from model import build_model


@torch.no_grad()
def predict(use_tta: bool = True):
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # ── Load model ────────────────────────────────────────────────
    model = build_model(pretrained=False).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded checkpoint from {CHECKPOINT_PATH}")

    # ── Choose transforms ─────────────────────────────────────────
    if use_tta:
        transforms_list = get_tta_transforms()
        print(f"[INFO] TTA enabled with {len(transforms_list)} views")
    else:
        transforms_list = [get_test_transforms()]

    # ── Accumulate soft predictions ───────────────────────────────
    base_ds = FoodTestDataset(TEST_DIR, transform=get_test_transforms())
    fnames = base_ds.fnames
    n_test = len(fnames)
    avg_probs = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)

    for t_idx, tfm in enumerate(transforms_list):
        ds = FoodTestDataset(TEST_DIR, transform=tfm)
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=False,
        )
        offset = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            bs = probs.shape[0]
            avg_probs[offset:offset + bs] += probs
            offset += bs
        print(f"  TTA view {t_idx + 1}/{len(transforms_list)} done")

    avg_probs /= len(transforms_list)
    preds_0based = avg_probs.argmax(axis=1)

    # Convert back to 1-based labels for submission
    preds_1based = preds_0based + LABEL_OFFSET

    # ── Build submission (match sample.csv format) ────────────────
    df = pd.DataFrame({
        "img_name": fnames,
        "label": preds_1based,
    })
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"[DONE] Submission saved to {SUBMISSION_PATH}  ({len(df)} rows)")
    print(df.head(10))


if __name__ == "__main__":
    predict(use_tta=True)
