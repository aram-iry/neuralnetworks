"""
Central configuration file - every hyperparameter lives here.
Food Recognition Challenge 2026 - 80 classes, flat image dirs, CSV labels.
Hardware target: NVIDIA GeForce GTX 1660 Super (6 GB VRAM),
                 AMD Ryzen 7 3700X (8 cores / 16 threads), 32 GB DDR4, Linux.
"""

import os
import torch

# --- Reproducibility --------------------------------------------------
SEED = 42

# --- Paths ------------------------------------------------------------
DATA_DIR = os.path.join("data")
TRAIN_DIR = os.path.join(DATA_DIR, "train_set", "train_set", "train_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set", "test_set", "test_set")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
CLASS_LIST_FILE = os.path.join(DATA_DIR, "class_list_food.txt")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample.csv")
OUTPUT_DIR = os.path.join("outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# --- Class mapping (parsed from class_list_food.txt) -----------------
# Labels in CSV are 1-based (1..80). We keep them as-is for submission.
NUM_CLASSES = 80
LABEL_OFFSET = 1  # CSV labels start at 1; internal indices = label - 1

# --- Device -----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image ------------------------------------------------------------
IMG_SIZE = 224          # Standard ResNet input size; better feature extraction
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]

# --- Training (tuned for GTX 1660 Super 6 GB + Ryzen 7 3700X) -------
BATCH_SIZE = 32         # Safe for 6 GB VRAM with AMP; use grad accumulation if needed
NUM_WORKERS = 6         # 6 workers provide good I/O throughput on an 8-core Ryzen 7 3700X
VAL_SPLIT = 0.15
EPOCHS = 25
EARLY_STOP_PATIENCE = 6

# --- Gradient accumulation -------------------------------------------
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS (=32 by default).
# Increase GRAD_ACCUM_STEPS (e.g. 2) to simulate larger batches without
# raising VRAM usage.
GRAD_ACCUM_STEPS = 1

# --- Optimizer / Scheduler -------------------------------------------
BACKBONE_LR = 5e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
SCHEDULER = "cosine"    # "cosine" | "step"
STEP_SIZE = 7           # only used when SCHEDULER == "step"
STEP_GAMMA = 0.1

# --- Mixed-precision -------------------------------------------------
# AMP halves VRAM usage and speeds up training on Turing GPUs (e.g. 1660 Super).
USE_AMP = torch.cuda.is_available()

# --- Fine-tuning strategy --------------------------------------------
FREEZE_BACKBONE_EPOCHS = 5   # train only head for first N epochs
