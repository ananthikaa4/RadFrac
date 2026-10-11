"""Inference service: load a trained model once and predict on new images.

Used by the demo app and the integration tests. Centralises model loading and
guarantees the exact same preprocessing as training (via ``preprocessing``),
so there is no train/serve skew and no hardcoded / fabricated predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .preprocessing import load_image_array, preprocess_for_dl


@dataclass
class Prediction:
    label: str              # predicted class name
    probability: float      # P(fractured)
    model_key: str          # which model produced it


class ModelUnavailableError(RuntimeError):
    """Raised when no trained model artifact can be loaded."""


class FracturePredictor:
    """Loads a single trained model and serves predictions.

    Deep models ("cnn"/"transfer") are loaded via Keras; the "classical" model
    is the pickled sklearn pipeline. The model is loaded once and reused.
    """

    def __init__(self, model_key: str = "classical"):
        if model_key not in config.MODEL_KEYS:
            raise ValueError(f"Unknown model_key {model_key!r}; "
                             f"choose from {config.MODEL_KEYS}")
        self.model_key = model_key
        self._model = None
        self._load()

    # -- loading ---------------------------------------------------------- #
    def _artifact_path(self) -> Path:
        return {
            "cnn": config.CNN_MODEL_FILE,
            "transfer": config.TRANSFER_MODEL_FILE,
            "classical": config.CLASSICAL_MODEL_FILE,
        }[self.model_key]

    def _load(self) -> None:
        path = self._artifact_path()
        if not path.exists():
            raise ModelUnavailableError(
                f"Trained artifact for '{self.model_key}' not found at {path}. "
                f"Run 'python -m src.train --model {self.model_key}' first."
            )
        if self.model_key in ("cnn", "transfer"):
            import tensorflow as tf
            self._model = tf.keras.models.load_model(path)
        else:
            from .model_svm import load_classical
            self._model = load_classical()

    # -- prediction ------------------------------------------------------- #
    def predict_array(self, img_uint8: np.ndarray) -> Prediction:
        """Predict from an RGB uint8 image array (already any size)."""
        # Crop + resize via the SAME helper the training pipeline uses, so an
        # uploaded image cannot skip the central crop.
        from .preprocessing import prepare_model_input
        img_uint8 = prepare_model_input(img_uint8)

        if self.model_key in ("cnn", "transfer"):
            x = preprocess_for_dl(img_uint8)[None, ...]
            prob = float(self._model.predict(x, verbose=0).reshape(-1)[0])
        else:
            from .features_classical import extract_features
            from .model_svm import predict_proba_positive
            feats = extract_features(img_uint8)[None, ...]
            prob = float(predict_proba_positive(self._model, feats)[0])

        from .thresholds import get_threshold
        threshold = get_threshold(self.model_key)
        label = (config.POSITIVE_CLASS if prob >= threshold
                 else config.CLASS_NAMES[1 - config.POSITIVE_INDEX])
        return Prediction(label=label, probability=prob, model_key=self.model_key)

    def predict_file(self, filepath: str) -> Prediction:
        return self.predict_array(load_image_array(filepath))


def artifact_paths() -> dict:
    """Model key -> artifact path, resolved at CALL time.

    Resolved on each call rather than captured in a module-level dict, so tests
    (and env-var overrides) that repoint ``config`` take effect. A frozen dict
    would silently keep pointing at the original paths.
    """
    return {
        "cnn": config.CNN_MODEL_FILE,
        "transfer": config.TRANSFER_MODEL_FILE,
        "classical": config.CLASSICAL_MODEL_FILE,
    }


def available_models(include_archived: bool = False) -> list[str]:
    """Model keys with a trained artifact that are ACTIVE in the application.

    The custom CNN is archived (see ``config.ACTIVE_MODEL_KEYS``) and is excluded
    unless explicitly requested, so it is never loaded or executed during normal
    inference.
    """
    keys = (config.MODEL_KEYS if include_archived else config.ACTIVE_MODEL_KEYS)
    paths = artifact_paths()
    return [k for k in keys if paths[k].exists()]


def load_comparison_metrics() -> dict | None:
    """Return the saved comparison metrics, or None if evaluation hasn't run."""
    if config.COMPARISON_METRICS_FILE.exists():
        return json.loads(config.COMPARISON_METRICS_FILE.read_text())
    return None
