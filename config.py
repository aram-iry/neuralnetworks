import os

# ─── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ─── Paths ───────────────────────────────────────────────────────
DATA_DIR = "data"
# Using your exact nested path for the flat directory:
TRAIN_DIR = os.path.join(DATA_DIR, "train_set", "train_set", "train_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set", "test_set", "test_set")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample.csv")

OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ─── Class mapping ───────────────────────────────────────────────
NUM_CLASSES = 80
LABEL_OFFSET = 1  # CSV labels start at 1, PyTorch needs them to start at 0

# ─── Image ─────────────────────────────���──────────────────────────
IMG_SIZE = 128          
IMG_MEAN = [0.485, 0.456, 0.406]   
IMG_STD = [0.229, 0.224, 0.225]

# ─── Training ─────────────────────────────────────────────────────
BATCH_SIZE = 64         
NUM_WORKERS = 8
VAL_SPLIT = 0.2        # 100/0 Split for the final push!
EPOCHS = 75
EARLY_STOP_PATIENCE = 30

# ─── Optimizer / Scheduler ────────────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
COSINE_T0 = 10           # Number of iterations for the first restart
COSINE_T_MULT = 2        
