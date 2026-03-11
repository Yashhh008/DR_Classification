"""
Model 1 – CNN Baseline for DR Classification.
Supports EfficientNet-B0 or ResNet-50 backbones with a custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """
    Standard CNN classifier using a pre-trained backbone.

    Architecture:
        Backbone → Global Average Pooling → Dropout → FC → Softmax
    """

    def __init__(self, num_classes: int = 5, backbone: str = "efficientnet_b0",
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            feature_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps before the global pooling layer
        (used by Grad-CAM)."""
        if self.backbone_name == "efficientnet_b0":
            return self.backbone.features(x)
        elif self.backbone_name == "resnet50":
            b = self.backbone
            x = b.conv1(x)
            x = b.bn1(x)
            x = b.relu(x)
            x = b.maxpool(x)
            x = b.layer1(x)
            x = b.layer2(x)
            x = b.layer3(x)
            x = b.layer4(x)
            return x
