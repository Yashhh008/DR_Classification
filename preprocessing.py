"""
Image preprocessing utilities for retinal fundus images.
Includes CLAHE enhancement, cropping, and normalization.
"""

import cv2
import numpy as np


def crop_black_borders(image: np.ndarray, tolerance: int = 10) -> np.ndarray:
    """Remove black background borders around the retinal fundus image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the green channel (most informative for DR) and recombine."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def ben_graham_preprocessing(image: np.ndarray, sigma: int = 10) -> np.ndarray:
    """Ben Graham's preprocessing: local average color subtraction.
    Popular baseline for fundus image enhancement in Kaggle competitions."""
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def preprocess_image(image: np.ndarray, size: tuple = (224, 224),
                     use_clahe: bool = True,
                     clahe_clip: float = 2.0,
                     clahe_grid: tuple = (8, 8)) -> np.ndarray:
    """Full preprocessing pipeline for a single fundus image.

    Steps:
        1. Crop black borders
        2. Resize to target size
        3. Optionally apply CLAHE
    """
    image = crop_black_borders(image)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        image = apply_clahe(image, clip_limit=clahe_clip,
                            grid_size=clahe_grid)
    return image
