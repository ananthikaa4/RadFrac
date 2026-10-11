"""Hybrid feature representation for the gradient-boosting models.

Each image becomes one vector made of two independent views:

* **Hand-crafted** - HOG + Canny edge-density grid (1,780-d). This is exactly
  ``features_classical.extract_features``, so the boosters see the same
  gradient/edge evidence as the existing SVM.
* **Deep** - the 1,280-d global-average-pooled embedding of MobileNetV2. It is
  read from the backbone inside the already-trained ``transfer.keras`` model.
  That backbone was frozen during training, so this is the ImageNet
  representation; reusing the saved model means no weight download at serve
  time and guarantees the embedding matches what the transfer model sees.

Feature blocks are cached per split under ``outputs/features`` so tuning runs
never recompute the expensive part.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .features_classical import extract_features
from .preprocessing import Sample, load_image_array, prepare_model_input, preprocess_for_dl

HANDCRAFTED_DIM = 1780          # 1,764 HOG + 16 Canny-grid cells at 128x128
DEEP_DIM = 1280                 # MobileNetV2 GAP output
FAMILIES = ("hog", "edges", "deep")


def family_slices(use_handcrafted: bool = True, use_deep: bool = True) -> dict[str, slice]:
    """Column ranges of each feature family inside the hybrid vector."""
    out: dict[str, slice] = {}
    pos = 0
    if use_handcrafted:
        out["hog"] = slice(0, HANDCRAFTED_DIM - 16)
        out["edges"] = slice(HANDCRAFTED_DIM - 16, HANDCRAFTED_DIM)
        pos = HANDCRAFTED_DIM
    if use_deep:
        out["deep"] = slice(pos, pos + DEEP_DIM)
    return out


# --------------------------------------------------------------------------- #
# Path handling
# --------------------------------------------------------------------------- #
def resolve_path(filepath: str) -> str:
    """Make a manifest path usable on this machine / inside the container.

    The split manifests were written on another machine with absolute Windows
    paths (``D:\\Old Laptop\\...\\raw\\<class>\\<file>``). If that path does
    not exist, fall back to ``config.RAW_DIR / <class> / <file>``.
    """
    if Path(filepath).exists():
        return filepath
    parts = filepath.replace("\\", "/").split("/")
    candidate = config.RAW_DIR / parts[-2] / parts[-1]
    return str(candidate)


def resolve_samples(samples: list[Sample]) -> list[Sample]:
    return [Sample(resolve_path(s.filepath), s.label) for s in samples]


# --------------------------------------------------------------------------- #
# Deep embedding model
# --------------------------------------------------------------------------- #
def build_embedding_model(transfer_model):
    """Sub-model of the trained transfer network that outputs the GAP vector."""
    import tensorflow as tf

    gap = None
    for layer in transfer_model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap = layer
    if gap is None:
        raise ValueError("transfer model has no GlobalAveragePooling2D layer")
    return tf.keras.Model(transfer_model.inputs, gap.output, name="radfrac_embedding")


def load_transfer_model():
    import tensorflow as tf

    path = config.TRANSFER_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"transfer model not found at {path}")
    return tf.keras.models.load_model(path)


# --------------------------------------------------------------------------- #
# Single image (serving path)
# --------------------------------------------------------------------------- #
def hybrid_vector(img_uint8: np.ndarray, embedding_model=None,
                  use_handcrafted: bool = True, use_deep: bool = True) -> np.ndarray:
    """Hybrid feature vector for one RGB uint8 image of any size.

    Uses ``prepare_model_input`` - the same crop + resize as training - so a
    served image cannot diverge from the cached training features.
    """
    small = prepare_model_input(img_uint8)
    blocks = []
    if use_handcrafted:
        blocks.append(extract_features(small))
    if use_deep:
        if embedding_model is None:
            raise ValueError("deep features requested but no embedding model given")
        x = preprocess_for_dl(small)[None, ...]
        blocks.append(np.asarray(embedding_model(x, training=False)).reshape(-1))
    return np.concatenate(blocks).astype("float32")


# --------------------------------------------------------------------------- #
# Batch extraction with caching (training path)
# --------------------------------------------------------------------------- #
def _cache_file(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"hybrid_{split}.npz"


def extract_split(samples: list[Sample], split: str, cache_dir: Path,
                  transfer_model=None, batch_size: int = 256,
                  force: bool = False) -> dict:
    """Return a dict with handcrafted / deep blocks, labels, paths and the
    transfer model's own probability for every image of one split.

    The transfer probability is stored too so the benchmark can score the
    existing MobileNetV2 model on exactly the same images without TensorFlow.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _cache_file(cache_dir, split)
    if out_path.exists() and not force:
        d = np.load(out_path, allow_pickle=False)
        return {k: d[k] for k in d.files}

    samples = resolve_samples(samples)
    missing = [s.filepath for s in samples if not Path(s.filepath).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}")

    if transfer_model is None:
        transfer_model = load_transfer_model()
    emb_model = build_embedding_model(transfer_model)

    hand, deep, tprob = [], [], []
    n = len(samples)
    for start in range(0, n, batch_size):
        chunk = samples[start:start + batch_size]
        imgs = [load_image_array(s.filepath) for s in chunk]
        hand.extend(extract_features(im) for im in imgs)
        x = np.stack([preprocess_for_dl(im) for im in imgs])
        deep.append(np.asarray(emb_model.predict(x, verbose=0)))
        tprob.append(np.asarray(transfer_model.predict(x, verbose=0)).reshape(-1))
        print(f"  [{split}] {min(start + batch_size, n)}/{n}", flush=True)

    data = {
        "handcrafted": np.vstack(hand).astype("float32"),
        "deep": np.vstack(deep).astype("float32"),
        "y": np.asarray([s.label for s in samples], dtype="int64"),
        "paths": np.asarray([s.filepath for s in samples]),
        "transfer_prob": np.concatenate(tprob).astype("float32"),
    }
    np.savez_compressed(out_path, **data)
    return data


def assemble(data: dict, use_handcrafted: bool = True, use_deep: bool = True) -> np.ndarray:
    blocks = []
    if use_handcrafted:
        blocks.append(data["handcrafted"])
    if use_deep:
        blocks.append(data["deep"])
    return np.hstack(blocks)
