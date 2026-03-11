"""
Utility helpers: model factory, seeding, device selection, model summary.
"""

import random
import os

import numpy as np
import torch

from config import Config
from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn_transformer import HybridCNNTransformer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


def build_model(cfg: Config) -> torch.nn.Module:
    """Instantiate the model specified in config."""
    if cfg.model_type == "cnn_baseline":
        model = CNNBaseline(
            num_classes=cfg.num_classes,
            backbone=cfg.cnn_backbone,
            pretrained=cfg.pretrained,
            dropout=cfg.classifier_dropout,
        )
    elif cfg.model_type == "hybrid_cnn_transformer":
        model = HybridCNNTransformer(
            num_classes=cfg.num_classes,
            backbone=cfg.cnn_backbone,
            pretrained=cfg.pretrained,
            freeze_backbone=cfg.freeze_backbone,
            embed_dim=cfg.transformer_embed_dim,
            num_heads=cfg.transformer_num_heads,
            num_layers=cfg.transformer_num_layers,
            mlp_dim=cfg.transformer_mlp_dim,
            transformer_dropout=cfg.transformer_dropout,
            classifier_dropout=cfg.classifier_dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    return model


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "frozen": total - trainable}


def print_model_summary(model: torch.nn.Module, cfg: Config):
    """Print a brief model summary."""
    params = count_parameters(model)
    print("\n" + "=" * 50)
    print(f"Model        : {cfg.model_type}")
    print(f"Backbone     : {cfg.cnn_backbone}")
    print(f"Total params : {params['total']:,}")
    print(f"Trainable    : {params['trainable']:,}")
    print(f"Frozen       : {params['frozen']:,}")
    print("=" * 50 + "\n")
