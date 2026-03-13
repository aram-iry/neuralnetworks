"""
Model factory – MobileNetV3-Small for fast CPU training.
Only ~2.5M params → trains well on CPU.
"""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    """
    MobileNetV3-Small: lightweight, fast on CPU, ImageNet-pretrained.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

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
