"""Data loading, splitting and image preprocessing for RadFrac.

The preprocessing here defines the single source of truth for how an image is
turned into model input. Both training and inference (the demo app) call
``load_image_array`` / ``preprocess_for_dl`` so the pipeline is guaranteed
consistent and there is no train/serve skew.

Layout expected on disk::

    data/raw/
        fractured/*.jpg
        not_fractured/*.jpg

Splits are persisted to ``data/splits/{train,val,test}.csv`` as
``filepath,label`` manifests so every model trains and is evaluated on exactly
the same partitions (prevents data leakage and makes runs reproducible).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config


@dataclass(frozen=True)
class Sample:
    filepath: str
    label: int          # integer class index (see config.CLASS_NAMES)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_samples(raw_dir: Path | str | None = None) -> list[Sample]:
    """Scan the directory-per-class layout and return a list of Samples."""
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    samples: list[Sample] = []
    for label, class_name in enumerate(config.CLASS_NAMES):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in config.VALID_EXTENSIONS:
                samples.append(Sample(str(path), label))
    return samples


def load_from_csv(
    csv_path: Path | str,
    image_dir: Path | str,
    filename_col: str,
    label_col: str,
    positive_value: str = "1",
) -> list[Sample]:
    """Alternative loader for CSV-manifest datasets (e.g. FracAtlas).

    ``label_col`` is compared as a string against ``positive_value`` to decide
    the positive (fractured) class; everything else is treated as negative.
    """
    image_dir = Path(image_dir)
    samples: list[Sample] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = row[filename_col]
            is_pos = str(row[label_col]).strip() == str(positive_value)
            label = config.POSITIVE_INDEX if is_pos else (1 - config.POSITIVE_INDEX)
            samples.append(Sample(str(image_dir / fname), label))
    return samples


# --------------------------------------------------------------------------- #
# Splitting (stratified, reproducible)
# --------------------------------------------------------------------------- #
def stratified_split(
    samples: list[Sample],
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified train/val/test split.

    The test set is carved out first and never touched again, then the
    validation set is carved from the remaining training pool. Stratification
    keeps the class ratio stable across all three partitions.
    """
    from sklearn.model_selection import train_test_split

    if not samples:
        return [], [], []

    labels = [s.label for s in samples]
    # Stratify only if every class has at least 2 members.
    can_stratify = min(np.bincount(labels)) >= 2 if len(set(labels)) > 1 else False
    strat = labels if can_stratify else None

    train_pool, test = train_test_split(
        samples, test_size=test_size, random_state=seed, stratify=strat
    )
    strat_pool = [s.label for s in train_pool] if can_stratify else None
    # val_size is a fraction of the ORIGINAL set; convert to fraction of pool.
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_pool, test_size=rel_val, random_state=seed, stratify=strat_pool
    )
    return train, val, test


# --------------------------------------------------------------------------- #
# Persisting splits
# --------------------------------------------------------------------------- #
def _write_manifest(path: Path, samples: Iterable[Sample]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filepath", "label"])
        for s in samples:
            writer.writerow([s.filepath, s.label])


def _read_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(Sample(row["filepath"], int(row["label"])))
    return samples


def save_splits(
    train: list[Sample], val: list[Sample], test: list[Sample],
    split_dir: Path | str | None = None,
) -> None:
    # Resolve at call time so tests (and env-var overrides) that patch
    # config.SPLIT_DIR take effect — a default-arg binding would freeze the
    # path at import time and leak test writes into the real data folder.
    split_dir = Path(split_dir) if split_dir is not None else config.SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(split_dir / "train.csv", train)
    _write_manifest(split_dir / "val.csv", val)
    _write_manifest(split_dir / "test.csv", test)
    # Record synthetic status at split time so it is captured even if raw data
    # is later swapped out. Imported lazily to avoid a circular import.
    try:
        from .dataset_status import is_synthetic_data
        synthetic = bool(is_synthetic_data())
    except Exception:
        synthetic = None
    summary = {
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "class_names": config.CLASS_NAMES,
        "synthetic_data": synthetic,
    }
    with open(split_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def load_splits(
    split_dir: Path | str | None = None,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    split_dir = Path(split_dir) if split_dir is not None else config.SPLIT_DIR
    return (
        _read_manifest(split_dir / "train.csv"),
        _read_manifest(split_dir / "val.csv"),
        _read_manifest(split_dir / "test.csv"),
    )


def splits_exist(split_dir: Path | str | None = None) -> bool:
    split_dir = Path(split_dir) if split_dir is not None else config.SPLIT_DIR
    return all((split_dir / f"{name}.csv").exists() for name in ("train", "val", "test"))


# --------------------------------------------------------------------------- #
# Image loading / preprocessing  (single source of truth)
# --------------------------------------------------------------------------- #
def center_crop_array(img: np.ndarray, fraction: float | None = None) -> np.ndarray:
    """Crop the central ``fraction`` of an image (H, W[, C]).

    Removes the peripheral ring where this dataset's label-correlated
    acquisition artefacts live (rotation corners, padding, burned-in markers).
    See ``config.CENTER_CROP_FRACTION`` for the rationale.
    """
    f = config.CENTER_CROP_FRACTION if fraction is None else fraction
    if f >= 1.0:
        return img
    f = max(0.05, min(1.0, f))
    h, w = img.shape[:2]
    ch, cw = int(round(h * f)), int(round(w * f))
    top = (h - ch) // 2
    left = (w - cw) // 2
    return img[top:top + ch, left:left + cw]


def load_image_array(
    filepath: str,
    size: tuple[int, int] = config.IMAGE_SIZE,
    crop: bool = True,
) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array resized to ``size``.

    Uses PIL so behaviour is identical whether TensorFlow is available or not
    (the demo app and classical pipeline both rely on this). The central crop is
    applied BEFORE resizing, matching ``_decode_and_resize`` in the tf.data
    pipeline exactly - both must agree or training and serving diverge.
    """
    from PIL import Image

    with Image.open(filepath) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if crop:
        arr = center_crop_array(arr)
    with Image.fromarray(arr) as im2:
        im2 = im2.resize((size[1], size[0]))       # PIL wants (W, H)
        return np.asarray(im2, dtype=np.uint8)


def prepare_model_input(img_uint8: np.ndarray,
                        size: tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """Crop + resize an in-memory image exactly as the training pipeline does.

    Single entry point for callers that already hold pixels (the API, the
    Streamlit upload path, Grad-CAM). Keeping this in one place is what prevents
    an uploaded image from skipping the central crop that training applied.
    """
    from PIL import Image

    arr = center_crop_array(img_uint8)
    with Image.fromarray(arr).convert("RGB") as im:
        im = im.resize((size[1], size[0]))
        return np.asarray(im, dtype="uint8")


def preprocess_for_dl(img_uint8: np.ndarray) -> np.ndarray:
    """Scale a uint8 image to float32 in [0, 1] for the CNN / transfer models."""
    return img_uint8.astype("float32") / 255.0


def load_dl_input(filepath: str) -> np.ndarray:
    """Convenience: disk -> model-ready (1, H, W, C) float32 batch."""
    arr = preprocess_for_dl(load_image_array(filepath))
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------- #
# tf.data pipelines for the deep models
# --------------------------------------------------------------------------- #
def _decode_and_resize(path, label):
    import tensorflow as tf

    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=config.IMAGE_CHANNELS, expand_animations=False)
    # Central crop BEFORE resize, mirroring load_image_array so the tf.data and
    # PIL paths produce identical inputs (no train/serve skew).
    if config.CENTER_CROP_FRACTION < 1.0:
        img = tf.image.central_crop(img, config.CENTER_CROP_FRACTION)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([*config.IMAGE_SIZE, config.IMAGE_CHANNELS])
    return img, label


def _augment(img, label):
    import tensorflow as tf

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_tf_dataset(
    samples: list[Sample],
    batch_size: int = config.BATCH_SIZE,
    training: bool = False,
    shuffle: bool = None,
):
    """Build a batched, prefetched tf.data.Dataset from samples."""
    import tensorflow as tf

    if shuffle is None:
        shuffle = training
    paths = [s.filepath for s in samples]
    labels = [s.label for s in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(samples), 1), seed=config.RANDOM_SEED)
    ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(train: list[Sample]) -> dict[int, float]:
    """Inverse-frequency class weights to counter class imbalance."""
    labels = np.array([s.label for s in train])
    counts = np.bincount(labels, minlength=len(config.CLASS_NAMES))
    total = counts.sum()
    n_classes = (counts > 0).sum()
    weights: dict[int, float] = {}
    for idx, c in enumerate(counts):
        weights[idx] = float(total / (n_classes * c)) if c > 0 else 0.0
    return weights


# --------------------------------------------------------------------------- #
# CLI: build & persist splits
# --------------------------------------------------------------------------- #
def main() -> None:
    config.ensure_output_dirs()
    samples = discover_samples()
    if not samples:
        raise SystemExit(
            f"No images found under {config.RAW_DIR}. Expected "
            f"{config.RAW_DIR}/<class>/*.jpg. Run "
            f"'python -m scripts.make_synthetic_data' or add a real dataset."
        )
    train, val, test = stratified_split(samples)
    save_splits(train, val, test)
    labels = np.bincount([s.label for s in samples], minlength=len(config.CLASS_NAMES))
    print(f"Discovered {len(samples)} images: " +
          ", ".join(f"{config.CLASS_NAMES[i]}={labels[i]}" for i in range(len(labels))))
    print(f"Split -> train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Manifests written to {config.SPLIT_DIR}")


if __name__ == "__main__":
    main()
