"""Model 3 - classical baseline (HOG/Canny features + SVM or RandomForest).

Wrapped in a scikit-learn Pipeline (StandardScaler -> classifier) so the scaler
is fit on training data only and travels with the model when persisted, keeping
inference preprocessing identical to training and preventing leakage.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import config


def build_classical(model_type: str = config.CLASSICAL_MODEL_TYPE):
    """Return an unfitted sklearn Pipeline for the classical baseline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model_type = model_type.lower()
    if model_type == "svm":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,               # needed for ROC-AUC
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown CLASSICAL_MODEL_TYPE: {model_type!r}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def save_classical(pipeline, path: Path | str | None = None) -> None:
    path = Path(path) if path is not None else config.CLASSICAL_MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh)


def load_classical(path: Path | str | None = None):
    path = Path(path) if path is not None else config.CLASSICAL_MODEL_FILE
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict_proba_positive(pipeline, X: np.ndarray) -> np.ndarray:
    """Return P(fractured) for each row of X, regardless of class ordering."""
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    pos_col = classes.index(config.POSITIVE_INDEX)
    return proba[:, pos_col]
