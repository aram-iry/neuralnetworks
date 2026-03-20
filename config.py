import os
import torch_directml

# --- Reproducibility ---
SEED = 42

# --- Paths ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train_set", "train_set", "train_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set", "test_set", "test_set")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
CLASS_LIST_FILE = os.path.join(DATA_DIR, "class_list_food.txt")
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# --- Model & Training ---
NUM_CLASSES = 80
LABEL_OFFSET = 1 
IMG_SIZE = 128      
BATCH_SIZE = 64     # Good for 8GB VRAM
NUM_WORKERS = 0     # Essential for Windows stability
VAL_SPLIT = 0.15
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch_directml.device()