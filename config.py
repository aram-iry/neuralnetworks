"""
Central configuration file – every hyperparameter lives here.
Food Recognition Challenge 2026 – 80 classes, flat image dirs, CSV labels.
"""

import os

# ─── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ─── Paths ────────────────────────────────────────────��───────────
DATA_DIR = os.path.join("data")
TRAIN_DIR = os.path.join(DATA_DIR, "train_set", "train_set", "train_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set", "test_set", "test_set")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
CLASS_LIST_FILE = os.path.join(DATA_DIR, "class_list_food.txt")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample.csv")
OUTPUT_DIR = os.path.join("outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ─── Class mapping (parsed from class_list_food.txt) ─────────────
# Labels in CSV are 1-based (1..80). We keep them as-is for submission.
NUM_CLASSES = 80
LABEL_OFFSET = 1  # CSV labels start at 1; internal indices = label - 1

# ─── Image ────────────────────────────────────────────────────────
IMG_SIZE = 128          # Smaller for CPU training speed
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]

# ─── Training (tuned for CPU on laptop) ───────────────────────────
BATCH_SIZE = 64         # CPU can handle larger batches (no VRAM limit)
NUM_WORKERS = 4         # your Zen3 has 8 threads
VAL_SPLIT = 0.15        # 85/15 split — more training data for CPU
EPOCHS = 20
EARLY_STOP_PATIENCE = 5

# ─── Optimizer / Scheduler ────────────────────────────────────────
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER = "cosine"    # "cosine" | "step"
STEP_SIZE = 7           # only used when SCHEDULER == "step"
STEP_GAMMA = 0.1

# ─── Mixed-precision ───────────────��─────────────────────────────
USE_AMP = False         # No CUDA → AMP disabled

# ─── Fine-tuning strategy ────────────────────────────────────────
FREEZE_BACKBONE_EPOCHS = 3   # train only head for first N epochs
