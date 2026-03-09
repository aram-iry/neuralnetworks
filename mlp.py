from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config: Dict):

        super().__init__()

        model_config = config["model"]
        self.input_dim = model_config["input_dim"]
        hidden_units = model_config["hidden_units"]
        num_classes = model_config["num_classes"]
        dropout_rate = model_config["dropout_rate"]  
        
        layers = []
        in_features = self.input_dim

        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden

        # map to num_classes (e.g., 2 for binary because we are using CrossEntropyLoss)
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
