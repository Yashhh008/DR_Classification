# Diabetic Retinopathy Detection using Hybrid CNN + Transformer

Automated detection and severity grading (0–4) of Diabetic Retinopathy from retinal fundus images using a hybrid deep learning architecture that combines CNN local feature extraction with Vision Transformer global attention.

## Project Structure

```
DR_retinopathy/
├── config.py                       # Centralized hyperparameters and paths
├── preprocessing.py                # CLAHE, border cropping, Ben Graham preprocessing
├── dataset.py                      # PyTorch Dataset, DataLoaders, augmentation
├── models/
│   ├── __init__.py
│   ├── cnn_baseline.py             # Model 1: EfficientNet-B0 / ResNet-50 baseline
│   ├── transformer_blocks.py       # Patch embedding + Transformer encoder
│   └── hybrid_cnn_transformer.py   # Model 2: CNN → Transformer → Classifier
├── train.py                        # Training loop, early stopping, scheduling
├── evaluate.py                     # Accuracy, F1, Kappa, ROC-AUC, confusion matrix
├── visualize.py                    # Grad-CAM, training curves, sample predictions
├── utils.py                        # Model factory, seeding, device helpers
├── run_experiment.py               # Main entry point (CLI)
├── requirements.txt
└── README.md
```

## Dataset Setup

1. Download the **APTOS 2019 Blindness Detection** dataset from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data).
2. Place the data as follows:

```
data/aptos2019/
├── train.csv
└── train_images/
    ├── 000c1434d8d7.png
    ├── ...
```

The CSV must have columns: `id_code`, `diagnosis` (0–4).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the Hybrid CNN + Transformer (default)
```bash
python run_experiment.py
```

### Train the CNN Baseline
```bash
python run_experiment.py --model cnn_baseline
```

### Use ResNet-50 backbone
```bash
python run_experiment.py --backbone resnet50
```

### Custom training parameters
```bash
python run_experiment.py --epochs 50 --batch_size 16 --lr 3e-4
```

### Evaluate a saved checkpoint
```bash
python run_experiment.py --evaluate --checkpoint outputs/checkpoints/best_model.pth
```

### Generate Grad-CAM visualizations
```bash
python run_experiment.py --gradcam --checkpoint outputs/checkpoints/best_model.pth
```

## Architecture

### Model 1 – CNN Baseline
```
Input (224×224×3) → EfficientNet-B0/ResNet-50 → GAP → Dropout → FC(5) → Softmax
```

### Model 2 – Hybrid CNN + Transformer
```
Input (224×224×3)
  → CNN Backbone (feature extraction)    → (B, C, 7, 7)
  → Flatten to token sequence            → (B, 49, C)
  → Project to embed_dim                 → (B, 49, 256)
  → Prepend CLS token + positional emb.  → (B, 50, 256)
  → Transformer Encoder (4 layers)       → (B, 50, 256)
  → CLS token output                     → (B, 256)
  → Dropout → FC(5) → Softmax
```

## Preprocessing Pipeline

1. **Crop black borders** around the retinal fundus
2. **Resize** to 224×224
3. **CLAHE** contrast enhancement (on LAB L-channel)
4. **Augmentation** (training only): rotation, flips, color jitter
5. **Normalize** to ImageNet statistics

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision / Recall / F1 | Per-class and macro/weighted averages |
| Quadratic Weighted Kappa | Ordinal agreement (standard DR metric) |
| ROC-AUC | One-vs-rest, macro averaged |
| Confusion Matrix | Normalized and raw |

## Outputs

All outputs are saved to `outputs/`:

- `checkpoints/best_model.pth` – Best model weights
- `logs/history.json` – Per-epoch training metrics
- `logs/test_metrics.json` – Final test set metrics
- `figures/training_history.png` – Loss/accuracy/LR curves
- `figures/confusion_matrix.png` – Confusion matrix heatmap
- `figures/sample_predictions.png` – Sample predictions with Grad-CAM

## DR Severity Classes

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild | Mild nonproliferative DR |
| 2 | Moderate | Moderate nonproliferative DR |
| 3 | Severe | Severe nonproliferative DR |
| 4 | Proliferative DR | Proliferative diabetic retinopathy |
