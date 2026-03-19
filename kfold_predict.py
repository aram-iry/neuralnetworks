import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import from your existing files
from config import (
    TEST_DIR, SAMPLE_CSV, SUBMISSION_PATH, 
    IMG_SIZE, IMG_MEAN, IMG_STD, LABEL_OFFSET, BATCH_SIZE
)
from model import build_model

# ─── Test Dataset for TTA ────────────────────────────────────────
class TTATestDataset(Dataset):
    def __init__(self, sample_csv, test_dir):
        self.df = pd.read_csv(sample_csv)
        self.test_dir = test_dir
        
        # Base transform (Normalize, Resize)
        self.base_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
        
        # Flipped transform for TTA
        self.flip_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0), # Force flip
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx, 0])
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += '.jpg'
            
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Generate both normal and flipped versions
        img_normal = self.base_transform(image)
        img_flipped = self.flip_transform(image)
        
        # Return the original image name from the CSV (without forced .jpg if it wasn't there)
        original_name = str(self.df.iloc[idx, 0])
        
        return img_normal, img_flipped, original_name

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 1. Load the Test Data
    print("[INFO] Loading test dataset for TTA...")
    test_dataset = TTATestDataset(SAMPLE_CSV, TEST_DIR)
    # Batch size can be larger for inference since no gradients are stored
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
    
    # 2. Load all 5 Fold Models
    models = []
    print("[INFO] Loading all 5 fold models into memory...")
    for i in range(1, 5):
        m = build_model(pretrained=False).to(device)
        model_path = f"outputs/best_model_fold_{i}.pth"
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Could not find {model_path}! Did fold {i} finish training?")
            return
            
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.eval()
        models.append(m)
        print(f"  ✓ Loaded Fold {i}")
        
    results = []
    
    # 3. Predict using the Ensemble
    print("[INFO] Starting ensemble prediction...")
    with torch.no_grad():
        for imgs_normal, imgs_flipped, img_names in test_loader:
            imgs_normal = imgs_normal.to(device)
            imgs_flipped = imgs_flipped.to(device)
            
            # Start with an empty tensor of zeros to accumulate the predictions
            # Shape will be [batch_size, 80 classes]
            ensemble_logits = torch.zeros((imgs_normal.size(0), 80)).to(device)
            
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                for m in models:
                    # Add predictions for both normal and flipped images
                    ensemble_logits += m(imgs_normal)
                    ensemble_logits += m(imgs_flipped)
            
            # Average the logits (5 models * 2 images = 10 total predictions per image)
            avg_logits = ensemble_logits / 10.0
            
            # Get the final winning class
            preds = avg_logits.argmax(dim=1).cpu().numpy()
            
            # Shift the labels back (from 0-79 back to 1-80)
            final_labels = preds + LABEL_OFFSET
            
            # Store results
            for name, label in zip(img_names, final_labels):
                results.append({"image_name": name, "label": label})
                
    # 4. Save to CSV
    print(f"[INFO] Predictions complete. Saving to {SUBMISSION_PATH}...")
    df_sub = pd.DataFrame(results)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print("[INFO] Done! You are ready to submit to Kaggle.")

if __name__ == "__main__":
    main()
