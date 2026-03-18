"""
Model factory – Upgraded to ResNet-18 for higher accuracy from scratch.
"""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model() -> nn.Module:
    """
    ResNet-18: Great balance of speed and high capacity for 80 food classes.
    """
    # Enforce NO pre-trained weights per project rules
    model = models.resnet18(weights=None)

    # Replace classifier head (ResNet uses 'fc')
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, NUM_CLASSES),
    )
    return model




def get_optimizer(model: nn.Module, backbone_lr: float, head_lr: float,
                  weight_decay: float):
    """Differential learning rates: smaller for backbone, larger for head."""
    backbone_params = [
        p for n, p in model.named_parameters()
        if "fc" not in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "fc" in n and p.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    param_groups.append({"params": head_params, "lr": head_lr})

    return __import__("torch").optim.AdamW(param_groups, weight_decay=weight_decay)
