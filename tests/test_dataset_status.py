"""Tests for synthetic-data detection and split test-isolation."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src import config
from src import dataset_status as ds


def test_marker_detection(tiny_raw_dataset):
    # The fixture's neutral filenames must NOT be treated as synthetic on their
    # own - only the marker or an explicit "synthetic_" prefix should trigger it.
    ds.remove_marker()
    assert ds.is_synthetic_data() is False

    ds.write_marker(meta={"test": True})
    assert (config.RAW_DIR / config.SYNTHETIC_MARKER_FILE.name).exists()
    assert ds.is_synthetic_data() is True

    ds.remove_marker()
    assert ds.is_synthetic_data() is False


def test_synthetic_prefix_detected_without_marker(tiny_raw_dataset):
    """The generator's 'synthetic_' prefix is enough on its own."""
    rng = np.random.default_rng(5)
    ds.remove_marker()
    d = config.RAW_DIR / config.POSITIVE_CLASS
    for i in range(20):
        arr = rng.integers(0, 256, (24, 24, 3), dtype="uint8")
        Image.fromarray(arr).save(
            d / f"{config.SYNTHETIC_FILENAME_PREFIX}{config.POSITIVE_CLASS}_{i:04d}.png")
    assert ds.is_synthetic_data() is True


def test_real_ingest_naming_never_flagged(tmp_path, monkeypatch):
    """Regression guard: real images named by the ingest scripts (xr_*) must
    never be mistaken for synthetic output.

    An earlier version keyed the heuristic on '<class>_', which collided with
    the ingest naming and raised a false synthetic warning on genuine data.
    """
    raw = tmp_path / "raw"
    rng = np.random.default_rng(7)
    for cls in config.CLASS_NAMES:
        d = raw / cls
        d.mkdir(parents=True)
        for i in range(10):
            arr = rng.integers(0, 256, (24, 24, 3), dtype="uint8")
            Image.fromarray(arr).save(d / f"{config.REAL_IMAGE_PREFIX}{i:06d}.jpg")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SYNTHETIC_MARKER_FILE", raw / ".SYNTHETIC_DATA")
    assert ds.is_synthetic_data() is False


def test_real_looking_data_not_flagged(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    for class_name in config.CLASS_NAMES:
        d = raw / class_name
        d.mkdir(parents=True)
        for i in range(6):
            arr = rng.integers(0, 256, (32, 32, 3), dtype="uint8")
            # Real-world-style opaque filenames (no class prefix, no marker).
            Image.fromarray(arr).save(d / f"IMG_{class_name[:3]}{i:03d}.png")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SYNTHETIC_MARKER_FILE", raw / ".SYNTHETIC_DATA")
    assert ds.is_synthetic_data() is False


def test_banner_is_ascii():
    # Must be encodable on a Windows cp1252 console without error.
    ds.banner().encode("cp1252")


def test_save_splits_respects_patched_config(tiny_raw_dataset):
    """Guards the fixed default-arg leak: save_splits must write to the patched
    SPLIT_DIR, never the real data folder."""
    from src.preprocessing import discover_samples, save_splits, splits_exist, stratified_split

    samples = discover_samples()
    train, val, test = stratified_split(samples)
    save_splits(train, val, test)
    # Written under the temp split dir the fixture patched in.
    assert splits_exist()
    assert (config.SPLIT_DIR / "train.csv").exists()
    assert str(config.SPLIT_DIR).startswith(str(config.RAW_DIR.parent))


def test_cleanup_classifies_legacy_and_real_files(tmp_path, monkeypatch):
    """The cleanup script must catch legacy-named synthetic files (which a
    plain 'synthetic_*' glob misses) while never touching real 'xr_*' files."""
    import importlib

    raw = tmp_path / "raw"
    rng = np.random.default_rng(11)
    for cls in config.CLASS_NAMES:
        d = raw / cls
        d.mkdir(parents=True)
        # Legacy generator naming: <class>_NNNN.png (4-digit counter).
        for i in range(4):
            Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype="uint8")).save(
                d / f"{cls}_{i:04d}.png")
        # Current generator naming.
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype="uint8")).save(
            d / f"{config.SYNTHETIC_FILENAME_PREFIX}{cls}_0000.png")
        # Real ingested images.
        for i in range(3):
            Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype="uint8")).save(
                d / f"{config.REAL_IMAGE_PREFIX}{i:06d}.jpg")

    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SYNTHETIC_MARKER_FILE", raw / ".SYNTHETIC_DATA")

    mod = importlib.import_module("scripts.clean_synthetic_data")
    synthetic, real, marker = mod.classify_files(raw)

    assert marker is False
    for cls in config.CLASS_NAMES:
        # 4 legacy + 1 prefixed = 5 synthetic; 3 real must be preserved.
        assert len(synthetic[cls]) == 5, synthetic[cls]
        assert len(real[cls]) == 3
        assert all(p.name.startswith(config.REAL_IMAGE_PREFIX) for p in real[cls])
