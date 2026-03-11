"""
Main experiment runner.

Usage examples:
    # Train the hybrid CNN+Transformer (default)
    python run_experiment.py

    # Train the CNN baseline with ResNet-50
    python run_experiment.py --model cnn_baseline --backbone resnet50

    # Only evaluate a saved checkpoint
    python run_experiment.py --evaluate --checkpoint outputs/checkpoints/best_model.pth

    # Generate Grad-CAM visualizations from a checkpoint
    python run_experiment.py --gradcam --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import json
import os

import numpy as np
import torch

from config import Config
from dataset import prepare_dataloaders
from train import train_model, validate
from evaluate import compute_metrics, print_metrics
from visualize import (
    plot_training_history,
    plot_confusion_matrix,
    show_sample_predictions,
    visualize_gradcam,
)
from utils import set_seed, get_device, build_model, print_model_summary


def parse_args():
    p = argparse.ArgumentParser(description="DR Detection Experiment Runner")
    p.add_argument("--model", type=str, default="hybrid_cnn_transformer",
                   choices=["cnn_baseline", "hybrid_cnn_transformer"],
                   help="Model architecture to use")
    p.add_argument("--backbone", type=str, default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet50"],
                   help="CNN backbone")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no_clahe", action="store_true",
                   help="Disable CLAHE preprocessing")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--evaluate", action="store_true",
                   help="Only evaluate (skip training)")
    p.add_argument("--gradcam", action="store_true",
                   help="Generate Grad-CAM visualizations")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Override default data directory")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Build config ───────────────────────────────────────────────────
    cfg = Config()
    cfg.model_type = args.model
    cfg.cnn_backbone = args.backbone
    cfg.freeze_backbone = args.freeze_backbone
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.no_clahe:
        cfg.apply_clahe = False
    if args.data_dir:
        cfg.data_dir = args.data_dir
        cfg.train_csv = os.path.join(args.data_dir, "train.csv")
        cfg.train_images_dir = os.path.join(args.data_dir, "train_images")
    cfg.make_dirs()

    set_seed(cfg.random_seed)
    device = get_device()

    # ── Data ───────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_weights = \
        prepare_dataloaders(cfg)

    # ── Model ──────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    print_model_summary(model, cfg)

    # ── Train or load ──────────────────────────────────────────────────
    if not args.evaluate and not args.gradcam:
        print("\nStarting training...\n")
        model, history = train_model(model, train_loader, val_loader,
                                     cfg, device)

        # Save history
        history_path = os.path.join(cfg.log_dir, "history.json")
        serializable = {k: [float(v) for v in vals]
                        for k, vals in history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)

        # Plot training curves
        plot_training_history(history, save_dir=cfg.figures_dir)

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint,
                                         map_location=device,
                                         weights_only=True))

    # ── Evaluate on test set ───────────────────────────────────────────
    print("\nEvaluating on test set...")
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels, probs = validate(
        model, test_loader, criterion, device
    )

    metrics = compute_metrics(labels, preds, probs, cfg.num_classes)
    print_metrics(metrics, class_names=cfg.class_names)

    # Save metrics
    metrics_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in metrics.items()}
    metrics_path = os.path.join(cfg.log_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_save, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Confusion matrix plot
    plot_confusion_matrix(metrics["confusion_matrix"], cfg.class_names,
                          save_dir=cfg.figures_dir)

    # ── Grad-CAM visualizations ────────────────────────────────────────
    if args.gradcam or not args.evaluate:
        print("\nGenerating Grad-CAM visualizations...")
        show_sample_predictions(model, test_loader.dataset, cfg.class_names,
                                device, num_samples=8,
                                save_dir=cfg.figures_dir)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
