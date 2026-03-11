"""
Model 2 – Hybrid CNN + Transformer for DR Classification.

Architecture:
    Input Image
    → CNN Backbone (EfficientNet-B0 / ResNet-50) feature extractor
    → Feature-map tokens + positional embeddings
    → Transformer Encoder (multi-head self-attention × L layers)
    → CLS token
    → Dense Layer
    → Softmax Output (5 DR severity classes)
"""

import torch
import torch.nn as nn
from torchvision import models

from .transformer_blocks import TransformerEncoder


class HybridCNNTransformer(nn.Module):
    """Hybrid CNN + Vision Transformer for DR severity classification."""

    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 512,
        transformer_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        # ── CNN backbone (feature extractor only) ──────────────────────
        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.cnn_features = base.features        # output: (B, 1280, 7, 7)
            cnn_out_channels = 1280
            feature_map_size = 7  # for 224×224 input
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            # Everything up to (but not including) avgpool and fc
            self.cnn_features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            cnn_out_channels = 2048
            feature_map_size = 7
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.cnn_features.parameters():
                param.requires_grad = False

        num_patches = feature_map_size * feature_map_size  # 49

        # ── Transformer encoder ────────────────────────────────────────
        self.transformer = TransformerEncoder(
            in_channels=cnn_out_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=transformer_dropout,
            num_patches=num_patches,
        )

        # ── Classification head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_features(x)       # (B, C, H, W)
        cls_token = self.transformer(features) # (B, embed_dim)
        logits = self.classifier(cls_token)    # (B, num_classes)
        return logits

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return CNN spatial feature maps (for Grad-CAM)."""
        return self.cnn_features(x)
