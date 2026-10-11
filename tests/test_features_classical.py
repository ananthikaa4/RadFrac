"""Unit tests for classical feature extraction."""
from __future__ import annotations

import numpy as np

from src import config
from src.features_classical import build_feature_matrix, extract_features, feature_dim
from src.preprocessing import discover_samples


def test_extract_features_is_1d_and_finite():
    img = np.random.default_rng(0).integers(
        0, 256, (*config.IMAGE_SIZE, 3), dtype="uint8"
    )
    feats = extract_features(img)
    assert feats.ndim == 1
    assert feats.shape[0] == feature_dim()
    assert np.isfinite(feats).all()


def test_feature_extraction_is_deterministic():
    img = np.random.default_rng(1).integers(
        0, 256, (*config.IMAGE_SIZE, 3), dtype="uint8"
    )
    assert np.allclose(extract_features(img), extract_features(img))


def test_build_feature_matrix_shapes(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    X, y = build_feature_matrix(samples)
    assert X.shape[0] == len(samples)
    assert X.shape[1] == feature_dim()
    assert y.shape[0] == len(samples)
    assert set(np.unique(y)).issubset({0, 1})
