"""
Training loop with early stopping, learning-rate scheduling,
and per-epoch metric logging.
"""

import os
import time
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from config import Config
from evaluate import compute_metrics


def build_optimizer(model: nn.Module, cfg: Config):
    return AdamW(model.parameters(), lr=cfg.learning_rate,
                 weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg: Config):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    elif cfg.scheduler == "step":
        return StepLR(optimizer, step_size=cfg.step_lr_step_size,
                      gamma=cfg.step_lr_gamma)
    return None


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch; return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on val/test set; return loss, accuracy, all preds & labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (running_loss / total, correct / total,
            all_preds, all_labels, all_probs)


def train_model(model, train_loader, val_loader, cfg: Config, device):
    """Full training routine.

    Returns:
        model        – best model (by val loss)
        history      – dict with per-epoch metrics
    """
    cfg.make_dirs()

    # Loss function
    class_weights = None
    if cfg.use_class_weights:
        from dataset import compute_class_weights
        import numpy as np
        labels = []
        for _, lbl in train_loader.dataset:
            labels.append(lbl)
        class_weights = compute_class_weights(
            np.array(labels), cfg.num_classes
        ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    early_stop = EarlyStopping(patience=cfg.early_stopping_patience)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, criterion, device
        )

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs} │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
            f"LR: {lr:.2e} │ {elapsed:.1f}s"
        )

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        if scheduler is not None:
            scheduler.step()

        early_stop(val_loss)
        if early_stop.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_model_wts)
    return model, history
