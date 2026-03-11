"""
Evaluation metrics for multi-class DR classification.
Computes accuracy, precision, recall, F1, confusion matrix,
ROC-AUC, and Quadratic Weighted Kappa.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_probs=None, num_classes: int = 5):
    """Compute a comprehensive metrics dictionary.

    Args:
        y_true   : ground-truth labels (list / ndarray)
        y_pred   : predicted labels (list / ndarray)
        y_probs  : predicted probabilities (N, num_classes), optional
        num_classes: number of classes

    Returns:
        dict with all metric values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro",
                             zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred,
                                              average="weighted",
                                              zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=list(range(num_classes))),
    }

    # ROC-AUC (requires probability estimates)
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = float("nan")

    return metrics


def print_metrics(metrics: dict, class_names=None):
    """Pretty-print the metrics dictionary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy              : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro)     : {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)        : {metrics['recall_macro']:.4f}")
    print(f"  F1 Score (macro)      : {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted)  : {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted)     : {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (weighted)   : {metrics['f1_weighted']:.4f}")
    print(f"  Quadratic W. Kappa    : {metrics['kappa']:.4f}")
    if "roc_auc_ovr" in metrics:
        print(f"  ROC-AUC (OVR, macro)  : {metrics['roc_auc_ovr']:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    if class_names:
        header = "         " + "  ".join(f"{n:>6s}" for n in class_names)
        print(header)
        for i, row in enumerate(cm):
            vals = "  ".join(f"{v:6d}" for v in row)
            print(f"  {class_names[i]:>6s}  {vals}")
    else:
        print(cm)
    print("=" * 60 + "\n")
