"""
Model definitions for the Food Recognition Challenge (80 classes).
Provides a ResNet18 backbone with a custom classification head,
plus helper functions used by the training loop.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ResNet18(nn.Module):
    """ResNet18 backbone with a 2-layer classification head for NUM_CLASSES."""

    def __init__(self, pretrained: bool = True):
        super(ResNet18, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the original fully-connected layer; keep feature extractor
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # 2-layer head: BN -> Dropout -> Linear -> ReLU -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_CLASSES),
        )

        if not pretrained:
            self._kaiming_init()

    def _kaiming_init(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Factory / helpers used by train.py
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True) -> ResNet18:
    """Return an initialised ResNet18 model with NUM_CLASSES output logits.

    When *pretrained* is True the backbone is loaded with ImageNet weights and
    only the head is trainable until :func:`unfreeze_backbone` is called.
    """
    model = ResNet18(pretrained=pretrained)
    if pretrained:
        # Freeze backbone; head trains from scratch
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    return model


def unfreeze_backbone(model: ResNet18) -> None:
    """Unfreeze all backbone parameters for full fine-tuning."""
    for param in model.backbone.parameters():
        param.requires_grad = True


def get_optimizer(
    model: ResNet18,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with differential learning rates: lower LR for backbone, higher for head."""
    param_groups = [
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Legacy classes kept for backward compatibility
# ---------------------------------------------------------------------------

class CNNSmall(nn.Module):
    def __init__(self):
        super(CNNSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
