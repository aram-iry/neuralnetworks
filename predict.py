import os
import torch
import pandas as pd
import torchvision.transforms as T
from config import *
from model import build_model
from dataset import FoodDataset

def predict():
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    tfm = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_ds = FoodDataset(TEST_DIR, transform=tfm)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    fnames, preds = [], []
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs).argmax(1).cpu().numpy()
            preds.extend(out + LABEL_OFFSET)
            fnames.extend(names)

    pd.DataFrame({"img_name": fnames, "label": preds}).to_csv(SUBMISSION_PATH, index=False)
    print(f"Prediction complete. File saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    predict()