"""Classical computer-vision feature extraction for the baseline model.

Combines Histogram-of-Oriented-Gradients (HOG) with a compact summary of Canny
edge structure. These hand-crafted features feed the SVM / RandomForest
baseline (``model_svm.py``) so it can be compared fairly against the learned
CNN representations.

The extractor works on the SAME preprocessed images (RGB uint8, resized to
``config.IMAGE_SIZE``) that the deep models see, so the comparison is apples to
apples.
"""
from __future__ import annotations

import numpy as np

from . import config
from .preprocessing import Sample, load_image_array


def _to_gray(img_uint8: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to a float grayscale image in [0, 1]."""
    from skimage.color import rgb2gray

    if img_uint8.ndim == 2:
        return img_uint8.astype("float32") / 255.0
    return rgb2gray(img_uint8).astype("float32")


def extract_features(img_uint8: np.ndarray) -> np.ndarray:
    """Extract a 1-D feature vector (HOG + Canny edge summary) from one image."""
    from skimage.feature import canny, hog

    gray = _to_gray(img_uint8)

    hog_vec = hog(
        gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # Canny edge map -> compact structural summary (edge density per grid cell).
    edges = canny(gray, sigma=config.CANNY_SIGMA)
    grid = 4
    h, w = edges.shape
    cell_h, cell_w = h // grid, w // grid
    edge_summary = []
    for i in range(grid):
        for j in range(grid):
            block = edges[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            edge_summary.append(float(block.mean()))
    edge_vec = np.asarray(edge_summary, dtype="float32")

    return np.concatenate([hog_vec.astype("float32"), edge_vec])


def build_feature_matrix(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a list of samples into an (X, y) matrix for scikit-learn."""
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for s in samples:
        img = load_image_array(s.filepath)
        feats.append(extract_features(img))
        labels.append(s.label)
    if not feats:
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype="int64")
    return np.vstack(feats), np.asarray(labels, dtype="int64")


def feature_dim() -> int:
    """Return the length of the feature vector (probe with a blank image)."""
    blank = np.zeros((*config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype="uint8")
    return int(extract_features(blank).shape[0])
