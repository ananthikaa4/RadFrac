"""Tests for Grad-CAM explainability on the custom CNN."""
from __future__ import annotations

import numpy as np
import pytest

from src import config


def test_gradcam_heatmap_shape_and_range():
    tf = pytest.importorskip("tensorflow")
    from src.gradcam import compute_gradcam, overlay_heatmap
    from src.model_cnn import build_cnn

    model = build_cnn()
    # Build the model by calling it once (mimics a freshly loaded model).
    dummy = np.zeros((1, *config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype="float32")
    model.predict(dummy, verbose=0)

    img = np.random.rand(*config.IMAGE_SIZE, config.IMAGE_CHANNELS).astype("float32")
    heat = compute_gradcam(model, img)
    assert heat.shape == config.IMAGE_SIZE
    assert 0.0 <= float(heat.min()) and float(heat.max()) <= 1.0

    overlay = overlay_heatmap(img, heat)
    assert overlay.shape == (*config.IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8
