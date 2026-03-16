"""
Model factory – EfficientNet-B0 for better accuracy while still being CPU-trainable.
~5.3M params → good balance of accuracy and training speed.
"""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # Freeze backbone initially
    freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_optimizer(model: nn.Module, backbone_lr: float, head_lr: float,
                  weight_decay: float):
    """Differential learning rates: smaller for backbone, larger for head."""
    backbone_params = [
        p for n, p in model.named_parameters()
        if "classifier" not in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "classifier" in n and p.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    param_groups.append({"params": head_params, "lr": head_lr})

    return __import__("torch").optim.AdamW(param_groups, weight_decay=weight_decay)
