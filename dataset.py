"""
Dataset and DataLoader utilities for APTOS 2019 Diabetic Retinopathy dataset.
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter

from config import Config
from preprocessing import preprocess_image


class DRDataset(Dataset):
    """PyTorch Dataset for the APTOS 2019 DR dataset."""

    def __init__(self, dataframe: pd.DataFrame, images_dir: str,
                 cfg: Config, transform=None, is_training: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.cfg = cfg
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["id_code"]
        label = int(row["diagnosis"])

        img_path = os.path.join(self.images_dir, f"{img_name}.png")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply fundus-specific preprocessing
        image = preprocess_image(
            image,
            size=self.cfg.image_size,
            use_clahe=self.cfg.apply_clahe,
            clahe_clip=self.cfg.clahe_clip_limit,
            clahe_grid=self.cfg.clahe_grid_size,
        )

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(cfg: Config) -> transforms.Compose:
    """Augmentation pipeline for training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(cfg.aug_rotation_degrees),
        transforms.RandomHorizontalFlip(cfg.aug_horizontal_flip_p),
        transforms.RandomVerticalFlip(cfg.aug_vertical_flip_p),
        transforms.ColorJitter(
            brightness=cfg.aug_brightness,
            contrast=cfg.aug_contrast,
            saturation=cfg.aug_saturation,
        ),
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])


def get_val_transforms(cfg: Config) -> transforms.Compose:
    """Minimal transform for validation / test."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = counter.get(c, 1)
        weights.append(total / (num_classes * count))
    return torch.FloatTensor(weights)


def prepare_dataloaders(cfg: Config):
    """Load CSV, split, and return DataLoaders + class weights.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    df = pd.read_csv(cfg.train_csv)

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(cfg.val_ratio + cfg.test_ratio),
        stratify=df["diagnosis"],
        random_state=cfg.random_seed,
    )

    # Second split: val vs test
    relative_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["diagnosis"],
        random_state=cfg.random_seed,
    )

    print(f"Split sizes → Train: {len(train_df)}  Val: {len(val_df)}  "
          f"Test: {len(test_df)}")

    # Class weights from training set
    class_weights = compute_class_weights(
        train_df["diagnosis"].values, cfg.num_classes
    )

    # Datasets
    train_ds = DRDataset(train_df, cfg.train_images_dir, cfg,
                         transform=get_train_transforms(cfg),
                         is_training=True)
    val_ds = DRDataset(val_df, cfg.train_images_dir, cfg,
                       transform=get_val_transforms(cfg))
    test_ds = DRDataset(test_df, cfg.train_images_dir, cfg,
                        transform=get_val_transforms(cfg))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=cfg.pin_memory)

    return train_loader, val_loader, test_loader, class_weights
