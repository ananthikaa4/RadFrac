"""Unified evaluation and model comparison for RadFrac.

Loads whichever trained models exist, runs them on the SAME held-out test set,
computes a consistent set of classification metrics, writes a comparison JSON,
and renders confusion-matrix images plus a combined ROC-curve figure.

Usage::

    python -m src.evaluate
"""
from __future__ import annotations

import json

import numpy as np

from . import config
from .preprocessing import Sample, load_splits, splits_exist


# --------------------------------------------------------------------------- #
# Prediction helpers (one per model family) -> P(fractured) per test sample
# --------------------------------------------------------------------------- #
def _predict_dl(model_file, test: list[Sample]) -> np.ndarray:
    import tensorflow as tf

    from .preprocessing import make_tf_dataset

    model = tf.keras.models.load_model(model_file)
    ds = make_tf_dataset(test, training=False, shuffle=False)
    probs = model.predict(ds, verbose=0).reshape(-1)
    return probs


def _predict_classical(test: list[Sample]) -> np.ndarray:
    from .features_classical import build_feature_matrix
    from .model_svm import load_classical, predict_proba_positive

    pipeline = load_classical()
    X, _ = build_feature_matrix(test)
    return predict_proba_positive(pipeline, X)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(config.CLASS_NAMES))))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
        "threshold": threshold,
    }
    # ROC-AUC only defined when both classes present in the test set.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
    return metrics


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_confusion_matrix(cm, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASS_NAMES)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_roc_comparison(roc_data: dict, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y_true, y_prob) in roc_data.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import auc as _auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate_all() -> dict:
    config.ensure_output_dirs()
    if not splits_exist():
        raise SystemExit("No splits found. Run 'python -m src.preprocessing' first.")

    _, _, test = load_splits()
    if not test:
        raise SystemExit("Test split is empty.")
    y_true = np.array([s.label for s in test])

    results: dict[str, dict] = {}
    roc_data: dict = {}

    tasks = [
        ("cnn", config.CNN_MODEL_FILE, lambda: _predict_dl(config.CNN_MODEL_FILE, test)),
        ("transfer", config.TRANSFER_MODEL_FILE,
         lambda: _predict_dl(config.TRANSFER_MODEL_FILE, test)),
        ("classical", config.CLASSICAL_MODEL_FILE, lambda: _predict_classical(test)),
    ]

    from .thresholds import get_threshold

    for name, artifact, predictor in tasks:
        if not artifact.exists():
            print(f"[{name}] artifact missing ({artifact}); skipping.")
            continue
        print(f"[{name}] evaluating on {len(test)} test images ...")
        y_prob = predictor()
        m = compute_metrics(y_true, y_prob, threshold=get_threshold(name))
        results[name] = m
        roc_data[name] = (y_true, y_prob)
        _save_confusion_matrix(
            m["confusion_matrix"], f"{name} confusion matrix",
            config.FIGURES_DIR / f"confusion_{name}.png",
        )
        print(f"    acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
              f"roc_auc={m['roc_auc']}")

    if not results:
        raise SystemExit("No trained models found. Run 'python -m src.train' first.")

    if len(roc_data) >= 1:
        _save_roc_comparison(roc_data, config.FIGURES_DIR / "roc_comparison.png")

    # Pick the best model by ROC-AUC (fallback to F1 when AUC undefined).
    def _score(m):
        return m["roc_auc"] if m["roc_auc"] is not None else m["f1"]

    best = max(results, key=lambda k: _score(results[k]))

    # Detect synthetic data (from raw folder or the persisted split summary) so
    # the metrics are never presented as clinically meaningful.
    synthetic = _detect_synthetic()

    payload = {
        "class_names": config.CLASS_NAMES,
        "models": results,
        "best_model": best,
        "selection_metric": "roc_auc",
        "synthetic_data": synthetic,
    }
    with open(config.COMPARISON_METRICS_FILE, "w") as fh:
        json.dump(payload, fh, indent=2)

    from .dataset_status import banner

    if synthetic:
        print(banner())
    print(f"Best model by ROC-AUC: {best}")
    print(f"Comparison metrics -> {config.COMPARISON_METRICS_FILE}")
    print(f"Figures -> {config.FIGURES_DIR}")

    _print_table(results)
    if synthetic:
        # Repeat after the table so it can't be read in isolation as legitimate.
        print(banner())
    return payload


def _detect_synthetic() -> bool:
    """True if raw data looks synthetic, or the split summary recorded it as such."""
    from .dataset_status import is_synthetic_data

    if is_synthetic_data():
        return True
    summary_path = config.SPLIT_DIR / "summary.json"
    if summary_path.exists():
        try:
            return bool(json.loads(summary_path.read_text()).get("synthetic_data"))
        except Exception:
            return False
    return False


def _print_table(results: dict) -> None:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'model':<12}" + "".join(f"{c:>11}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<12}"
        for c in cols:
            v = m[c]
            row += f"{'   n/a' if v is None else f'{v:>11.3f}'}"
        print(row)


if __name__ == "__main__":
    evaluate_all()
