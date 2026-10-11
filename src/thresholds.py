"""Per-model decision-threshold selection and persistence.

Deep nets (especially small BatchNorm CNNs on small datasets) can rank samples
well while emitting poorly-calibrated probabilities, so a fixed 0.5 cut-off is
unreliable. We pick each model's operating threshold on the VALIDATION split by
maximising F1, then persist it. The test set is never used for this choice, so
evaluation stays unbiased.
"""
from __future__ import annotations

import json

import numpy as np

from . import config


def best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold in (0,1) that maximises F1 on the data.

    Falls back to ``DEFAULT_THRESHOLD`` when only one class is present.
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return config.DEFAULT_THRESHOLD

    candidates = np.unique(np.round(y_prob, 4))
    # Search midpoints so ties don't hug a single predicted probability.
    grid = np.clip(candidates, 0.01, 0.99)
    best_t, best_f1 = config.DEFAULT_THRESHOLD, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def load_thresholds() -> dict[str, float]:
    if config.THRESHOLDS_FILE.exists():
        # Guarded like load_cost_thresholds: a truncated thresholds.json from
        # an interrupted deploy would otherwise raise on every prediction.
        try:
            return json.loads(config.THRESHOLDS_FILE.read_text())
        except Exception:
            return {}
    return {}


def load_cost_thresholds() -> dict[str, float]:
    """Cost-optimal thresholds from scripts/threshold_analysis.py, if present."""
    if config.COST_THRESHOLDS_FILE.exists():
        try:
            return json.loads(config.COST_THRESHOLDS_FILE.read_text())
        except Exception:
            return {}
    return {}


def get_threshold_with_source(model_key: str) -> tuple[float, str]:
    """Return (threshold, source) for a model.

    Preference order, and why:

    1. ``cost`` - the weighted-cost-optimal point (FN weighted above FP). This
       is the clinically correct operating point for a screening tool, so it
       wins whenever the cost analysis has been run.
    2. ``f1``   - the F1-optimal point saved during training.
    3. ``default`` - 0.5.

    All of these are selected on validation data only.
    """
    cost = load_cost_thresholds()
    if model_key in cost:
        return float(cost[model_key]), "cost"
    f1 = load_thresholds()
    if model_key in f1:
        return float(f1[model_key]), "f1"
    return config.DEFAULT_THRESHOLD, "default"


def get_threshold(model_key: str) -> float:
    return get_threshold_with_source(model_key)[0]


def save_threshold(model_key: str, threshold: float) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_thresholds()
    data[model_key] = float(threshold)
    config.THRESHOLDS_FILE.write_text(json.dumps(data, indent=2))
