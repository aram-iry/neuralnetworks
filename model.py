import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES

def build_model(pretrained=False):
    # Load ResNet-18
    # Using 'weights=None' which is the modern PyTorch equivalent of pretrained=False
    model = models.resnet18(weights=None)
    
    # Replace the final fully connected layer to match our 80 classes
    # AND add 50% Dropout to prevent overfitting
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    return model
