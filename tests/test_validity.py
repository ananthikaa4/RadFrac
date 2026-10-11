"""Tests for the input-validity gate (4.8)."""
from __future__ import annotations

import numpy as np

from src import config
from src.validity import check_gradcam_coherence, check_image


def test_black_image_is_rejected():
    img = np.zeros((128, 128, 3), dtype="uint8")
    res = check_image(img)
    assert res.ok is False
    assert res.should_block is True
    assert any("black" in e for e in res.errors)


def test_white_image_is_rejected():
    img = np.full((128, 128, 3), 255, dtype="uint8")
    res = check_image(img)
    assert res.should_block is True


def test_flat_gray_image_rejected_for_no_contrast():
    img = np.full((128, 128, 3), 128, dtype="uint8")
    res = check_image(img)
    assert res.should_block is True
    assert any("contrast" in e for e in res.errors)


def test_plausible_xray_passes():
    rng = np.random.default_rng(0)
    # Dark background with a bright structure = plausible radiograph.
    img = np.full((160, 140, 3), 30, dtype="uint8")
    img[40:120, 50:90] = 200
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype("uint8")
    res = check_image(img)
    assert res.ok is True
    assert res.should_block is False


def test_colorful_image_warns():
    rng = np.random.default_rng(1)
    img = np.zeros((128, 128, 3), dtype="uint8")
    img[..., 0] = 220          # strong red channel only -> high saturation
    img[..., 1] = 40
    img[..., 2] = 30
    img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype("uint8")
    res = check_image(img)
    assert any("colour" in w or "color" in w for w in res.warnings)


def test_extreme_aspect_ratio_warns():
    rng = np.random.default_rng(2)
    img = np.clip(rng.normal(120, 40, (40, 400, 3)), 0, 255).astype("uint8")
    res = check_image(img)
    assert any("aspect ratio" in w for w in res.warnings)


def test_metrics_reported():
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(120, 40, (100, 100, 3)), 0, 255).astype("uint8")
    res = check_image(img)
    for k in ("mean_intensity", "intensity_std", "aspect_ratio", "color_saturation"):
        assert k in res.metrics


def test_gradcam_coherence_flags_empty_attention():
    heat = np.zeros((64, 64), dtype="float32")
    res = check_gradcam_coherence(heat)
    assert any("diffuse" in w or "empty" in w for w in res.warnings)


def test_gradcam_coherence_flags_flooded_attention():
    heat = np.ones((64, 64), dtype="float32")
    res = check_gradcam_coherence(heat)
    assert any("whole image" in w for w in res.warnings)


def test_gradcam_coherence_accepts_focal_attention():
    heat = np.zeros((64, 64), dtype="float32")
    heat[28:36, 28:36] = 1.0
    res = check_gradcam_coherence(heat)
    assert res.warnings == []
