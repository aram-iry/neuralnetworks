"""
Deterministic seeding for full reproducibility.
"""

import os
import random
import numpy as np
import torch

from config import SEED


def seed_everything(seed: int = SEED) -> None:
    """Seed all RNGs and force deterministic back-ends."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
