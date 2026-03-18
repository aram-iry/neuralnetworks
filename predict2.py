"""
Inference script to generate submission.csv for the leaderboard.
Includes Test-Time Augmentation (TTA) for a free accuracy boost!
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    TEST_DIR, CHECKPOINT_PATH, SUBMISSION_PATH, SAMPLE_CSV,
    IMG_SIZE, IMG_MEAN, IMG_STD, LABEL_OFFSET, BATCH_SIZE
)
from model import build_model


class TestDataset(Dataset):
    def __init__(self, test_dir, transform_normal, transform_flip):
        self.test_dir = test_dir
        self.transform_normal = transform_normal
        self.transform_flip = transform_flip
        
        # Grab all image files in the test directory
        self.images = sorted([img for img in os.listdir(test_dir) if img.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Return both the normal image and a flipped version for TTA
        img_normal = self.transform_normal(image)
        img_flipped = self.transform_flip(image)
        
        return img_normal, img_flipped, img_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ─── Transforms (Normal & TTA) ──────────────────────────────
    transform_normal = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])
    
    transform_flip = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0), # Force flip
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # ─── Data Loader ────────────────────────────────────────────
    test_dataset = TestDataset(TEST_DIR, transform_normal, transform_flip)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"[INFO] Found {len(test_dataset)} test images.")

    # ─── Load Model ─────────────────────────────────────────────
    # Must use pretrained=False to match the architecture we trained
    model = build_model().to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find weights at {CHECKPOINT_PATH}. Did training finish?")
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("[INFO] Successfully loaded best_model.pth")

    # ─── Inference Loop with TTA ────────────────────────────────
    predictions = []
    image_names = []

    print("[INFO] Generating predictions...")
    with torch.no_grad():
        for imgs_normal, imgs_flipped, img_names in test_loader:
            imgs_normal = imgs_normal.to(device)
            imgs_flipped = imgs_flipped.to(device)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                # Predict on both normal and flipped images
                logits_normal = model(imgs_normal)
                logits_flipped = model(imgs_flipped)
                
                # Average the confidence scores
                avg_logits = (logits_normal + logits_flipped) / 2.0
                
                # Get the highest confidence class
                preds = avg_logits.argmax(dim=1).cpu().numpy()

            # Add offset if your classes are 1-80 instead of 0-79
            preds = preds + LABEL_OFFSET
            
            predictions.extend(preds)
            image_names.extend(img_names)

    # ─── Create Submission CSV ──────────────────────────────────
    # Format according to the Kaggle/Evaluation standard
    submission_df = pd.DataFrame({
        'image': image_names,   # Check your sample.csv to ensure these header names match exactly!
        'label': predictions    # Sometimes they are 'Id' and 'Category'
    })

    # Optional: If you have a sample.csv, you can read it to match exact headers
    if os.path.exists(SAMPLE_CSV):
        sample = pd.read_csv(SAMPLE_CSV)
        submission_df.columns = sample.columns

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n[DONE] Saved predictions to {SUBMISSION_PATH}")
    print("[INFO] You are ready to submit to the leaderboard!")


if __name__ == "__main__":
    main()
