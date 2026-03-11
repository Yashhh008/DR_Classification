"""
Configuration module for Diabetic Retinopathy Detection Project.
Central place for all hyperparameters and paths.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_dir: str = os.path.join("data", "aptos2019")
    train_csv: str = os.path.join("data", "aptos2019", "train.csv")
    train_images_dir: str = os.path.join("data", "aptos2019", "train_images")
    output_dir: str = "outputs"
    checkpoint_dir: str = os.path.join("outputs", "checkpoints")
    log_dir: str = os.path.join("outputs", "logs")
    figures_dir: str = os.path.join("outputs", "figures")

    # ── Dataset ────────────────────────────────────────────────────────
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
    ])
    # Split ratios: train / val / test
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Preprocessing ──────────────────────────────────────────────────
    image_size: Tuple[int, int] = (224, 224)
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # ImageNet normalization
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ── Augmentation ───────────────────────────────────────────────────
    aug_rotation_degrees: int = 30
    aug_horizontal_flip_p: float = 0.5
    aug_vertical_flip_p: float = 0.5
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.2

    # ── Model ──────────────────────────────────────────────────────────
    # "cnn_baseline" | "hybrid_cnn_transformer"
    model_type: str = "hybrid_cnn_transformer"
    cnn_backbone: str = "efficientnet_b0"  # "efficientnet_b0" | "resnet50"
    freeze_backbone: bool = False
    pretrained: bool = True

    # Transformer hyper-parameters
    transformer_embed_dim: int = 256
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4
    transformer_mlp_dim: int = 512
    transformer_dropout: float = 0.1

    # Classification head
    classifier_dropout: float = 0.3

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine" | "step" | "none"
    step_lr_step_size: int = 10
    step_lr_gamma: float = 0.1
    early_stopping_patience: int = 7
    label_smoothing: float = 0.1
    use_class_weights: bool = True

    # ── Device ─────────────────────────────────────────────────────────
    num_workers: int = 4
    pin_memory: bool = True

    def make_dirs(self):
        """Create output directories if they don't exist."""
        for d in [self.output_dir, self.checkpoint_dir,
                  self.log_dir, self.figures_dir]:
            os.makedirs(d, exist_ok=True)
