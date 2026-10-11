"""Shared pytest fixtures for RadFrac tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


@pytest.fixture
def tiny_raw_dataset(tmp_path, monkeypatch):
    """Create a tiny on-disk directory-per-class dataset and point config at it.

    Yields the raw dir. Redirects config.RAW_DIR / SPLIT_DIR / output dirs into
    a temp location so tests never touch real data or artifacts.
    """
    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    for label, class_name in enumerate(config.CLASS_NAMES):
        d = raw / class_name
        d.mkdir(parents=True)
        for i in range(8):
            base = 40 if class_name == config.POSITIVE_CLASS else 180
            arr = np.clip(rng.normal(base, 20, (48, 48, 3)), 0, 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{class_name}_{i}.png")

    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SPLIT_DIR", tmp_path / "splits")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "SYNTHETIC_MARKER_FILE", raw / ".SYNTHETIC_DATA")
    monkeypatch.setattr(config, "CLASSICAL_MODEL_FILE", tmp_path / "models" / "classical.pkl")
    monkeypatch.setattr(config, "CNN_MODEL_FILE", tmp_path / "models" / "cnn.keras")
    monkeypatch.setattr(config, "TRANSFER_MODEL_FILE", tmp_path / "models" / "transfer.keras")
    monkeypatch.setattr(config, "THRESHOLDS_FILE", tmp_path / "models" / "thresholds.json")
    monkeypatch.setattr(config, "COMPARISON_METRICS_FILE", tmp_path / "metrics" / "comparison_metrics.json")
    return raw
