"""Model definition for pairwise pottery compatibility classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class PairClassifier(nn.Module):
    """Binary classifier for one exterior image and one interior image."""

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode one branch of pottery images."""
        return self.backbone(images)

    def forward(self, exterior: torch.Tensor, interior: torch.Tensor) -> torch.Tensor:
        """Predict one compatibility logit for each input pair."""
        exterior_features = self.encode(exterior)
        interior_features = self.encode(interior)
        fused = torch.cat(
            [
                exterior_features,
                interior_features,
                torch.abs(exterior_features - interior_features),
                exterior_features * interior_features,
            ],
            dim=1,
        )
        return self.classifier(fused).squeeze(1)
