"""
Visualization utilities:
  - Grad-CAM heatmap generation and overlay
  - Training history plots
  - Confusion matrix heatmap
  - Sample prediction display
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import Config


# ═══════════════════════════════════════════════════════════════════════
# Grad-CAM
# ═══════════════════════════════════════════════════════════════════════

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Works with any model that exposes a `get_feature_maps` method
    (both CNNBaseline and HybridCNNTransformer do).
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def generate(self, input_tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single image.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor.
            target_class: class index to visualise; if None, uses predicted.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward through CNN feature extractor
        feature_maps = self.model.get_feature_maps(input_tensor)
        feature_maps.retain_grad()

        # Forward through the rest of the model to get logits
        if hasattr(self.model, "transformer"):
            # Hybrid model
            cls_token = self.model.transformer(feature_maps)
            logits = self.model.classifier(cls_token)
        else:
            # CNN baseline
            pooled = F.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
            logits = self.model.classifier(pooled)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Grad-CAM weights
        gradients = feature_maps.grad[0]           # (C, H, W)
        weights = gradients.mean(dim=(1, 2))        # (C,)
        cam = (weights[:, None, None] * feature_maps[0].detach()).sum(dim=0)
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay(self, image_np: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
        """Overlay Grad-CAM heatmap on the original image.

        Args:
            image_np: (H, W, 3) uint8 RGB image.
            heatmap : (Hc, Wc) float in [0, 1].
            alpha   : blending factor.

        Returns:
            (H, W, 3) uint8 overlay image.
        """
        h, w = image_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap_colored + (1 - alpha) * image_np).astype(np.uint8)
        return overlay


def visualize_gradcam(model, image_tensor, original_image, class_names,
                      device, save_path=None):
    """Generate and display Grad-CAM for a single sample.

    Args:
        model         : trained model
        image_tensor  : (1, C, H, W) preprocessed tensor
        original_image: (H, W, 3) uint8 RGB numpy array
        class_names   : list of class name strings
        device        : torch device
        save_path     : if provided, save the figure
    """
    gc = GradCAM(model, device)

    # Get prediction
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = probs.argmax()
        confidence = probs[pred_class]

    heatmap = gc.generate(image_tensor, target_class=int(pred_class))
    overlay = gc.overlay(original_image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {class_names[pred_class]} ({confidence:.2%})"
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM saved to {save_path}")
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Training history plots
# ═══════════════════════════════════════════════════════════════════════

def plot_training_history(history: dict, save_dir: str = None):
    """Plot loss, accuracy, and learning-rate curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR
    axes[2].plot(epochs, history["lr"], color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to {path}")
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Confusion matrix
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, save_dir=None, normalize=True):
    """Plot a confusion matrix as a heatmap."""
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {path}")
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Sample predictions gallery
# ═══════════════════════════════════════════════════════════════════════

def show_sample_predictions(model, dataset, class_names, device,
                            num_samples: int = 8, save_dir=None):
    """Display a grid of sample images with predictions and Grad-CAM."""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))],
        std=[1.0 / s for s in (0.229, 0.224, 0.225)],
    )

    gc = GradCAM(model, device)
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        img_tensor, label = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            pred = logits.argmax(1).item()
            prob = torch.softmax(logits, 1)[0, pred].item()

        # Denormalize for display
        img_display = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display * 255, 0, 255).astype(np.uint8)

        heatmap = gc.generate(img_tensor.unsqueeze(0), target_class=pred)
        overlay = gc.overlay(img_display, heatmap)

        axes[r, c].imshow(overlay)
        color = "green" if pred == label else "red"
        axes[r, c].set_title(
            f"GT: {class_names[label]}\n"
            f"Pred: {class_names[pred]} ({prob:.0%})",
            fontsize=9, color=color,
        )
        axes[r, c].axis("off")

    # Hide unused axes
    for i in range(num_samples, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "sample_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Sample predictions saved to {path}")
    plt.show()
    plt.close()
