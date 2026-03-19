"""
Model factory – ResNet-34 with SE attention and concatenated pooling
for fine-grained food classification from scratch.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: learns per-channel attention weights."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


def _add_se_to_resnet(model, reduction=16):
    """Inject SE blocks after each BasicBlock/Bottleneck in a ResNet."""
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, name)
        for i, block in enumerate(layer):
            channels = block.conv2.out_channels
            block.se = SEBlock(channels, reduction)
            original_forward = block.forward

            def make_forward(blk):
                def forward_with_se(x):
                    identity = x
                    out = blk.conv1(x)
                    out = blk.bn1(out)
                    out = blk.relu(out)
                    out = blk.conv2(out)
                    out = blk.bn2(out)
                    out = blk.se(out)
                    if blk.downsample is not None:
                        identity = blk.downsample(x)
                    out += identity
                    out = blk.relu(out)
                    return out
                return forward_with_se

            block.forward = make_forward(block)


class ConcatPoolHead(nn.Module):
    """Concatenates AdaptiveAvgPool and AdaptiveMaxPool, then classifies."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        avg = self.avgpool(x).flatten(1)
        mx = self.maxpool(x).flatten(1)
        return self.head(torch.cat([avg, mx], dim=1))


def build_model() -> nn.Module:
    """
    ResNet-34 with SE attention and concat pooling for 80 food classes.
    """
    # Enforce NO pre-trained weights per project rules
    model = models.resnet34(weights=None)

    # Inject SE attention into every residual block
    _add_se_to_resnet(model, reduction=16)

    # Get the number of features before the original fc layer
    in_features = model.fc.in_features  # 512 for ResNet-34

    # Remove original avgpool + fc — we replace them with ConcatPoolHead
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()

    # Wrap into a sequential model
    class ResNetWithConcatPool(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            # Run through backbone up to the last conv layer
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            # x is now (B, 512, H, W) — pass to concat pool head
            return self.head(x)

    head = ConcatPoolHead(in_features, NUM_CLASSES)
    return ResNetWithConcatPool(model, head)




def get_optimizer(model: nn.Module, backbone_lr: float, head_lr: float,
                  weight_decay: float):
    """Differential learning rates: smaller for backbone, larger for head."""
    backbone_params = [
        p for n, p in model.named_parameters()
        if "head" not in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "head" in n and p.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    param_groups.append({"params": head_params, "lr": head_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
