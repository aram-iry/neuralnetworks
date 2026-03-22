import os
import argparse
import torch
import pandas as pd
import torchvision.transforms as T
from config import *
from model_msnet import build_model
from dataset import FoodDataset

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=IMG_SIZE,
                        help=f"Image size used during training (default: {IMG_SIZE})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for inference (default: {BATCH_SIZE})")
    args = parser.parse_args()

    img_size = args.img_size
    batch_size = args.batch_size
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{img_size}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. "
                                f"Have you trained with --img_size {img_size}?")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds = FoodDataset(TEST_DIR, transform=tfm)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                         num_workers=NUM_WORKERS)

    fnames, preds = [], []
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs).argmax(1).cpu().numpy()
            preds.extend(out + LABEL_OFFSET)
            fnames.extend(names)

    submission_path = os.path.join(OUTPUT_DIR, f"submission_{img_size}.csv")
    pd.DataFrame({"img_name": fnames, "label": preds}).to_csv(submission_path, index=False)
    print(f"[INFO] Predictions saved to {submission_path}")

if __name__ == "__main__":
    predict()
