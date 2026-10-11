"""Unified training entrypoint for RadFrac.

Usage::

    python -m src.train --model all
    python -m src.train --model cnn --epochs_cnn 20
    python -m src.train --model classical

Trains any subset of the three models on the persisted train/val splits and
writes artifacts to ``outputs/models/``. All models share the exact same splits
(built by ``src.preprocessing``) so downstream evaluation is fair and free of
leakage.
"""
from __future__ import annotations

import argparse
import json
import time

from . import config
from .preprocessing import (
    Sample,
    compute_class_weights,
    discover_samples,
    load_splits,
    save_splits,
    splits_exist,
    stratified_split,
)


def _save_history(model_key: str, hist, epochs_requested: int) -> None:
    """Persist the Keras history so epoch counts and curves are auditable later.

    Records whether early stopping fired, which is the difference between
    "converged" and "stopped prematurely at a poor value".
    """
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    ran = len(next(iter(h.values()))) if h else 0
    payload = {
        "model": model_key,
        "epochs_requested": int(epochs_requested),
        "epochs_ran": ran,
        "early_stopped": bool(ran < epochs_requested),
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "center_crop_fraction": config.CENTER_CROP_FRACTION,
        "final": {k: vals[-1] for k, vals in h.items()},
        "best_val_auc": max(h.get("val_auc", [float("nan")])),
        "history": h,
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.METRICS_DIR / f"train_history_{model_key}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[{model_key}] history ({ran}/{epochs_requested} epochs, "
          f"early_stopped={payload['early_stopped']}) -> {out.name}")


def _tune_dl_threshold(model_key: str, model, val: list[Sample]) -> None:
    """Pick and persist the val-set F1-optimal threshold for a deep model."""
    import numpy as np

    from .preprocessing import make_tf_dataset
    from .thresholds import best_threshold_f1, save_threshold

    if not val:
        return
    y_true = np.array([s.label for s in val])
    val_ds = make_tf_dataset(val, training=False, shuffle=False)
    y_prob = model.predict(val_ds, verbose=0).reshape(-1)
    t = best_threshold_f1(y_true, y_prob)
    save_threshold(model_key, t)
    print(f"[{model_key}] tuned decision threshold on val -> {t:.3f}")


def _ensure_splits() -> tuple[list[Sample], list[Sample], list[Sample]]:
    if splits_exist():
        return load_splits()
    samples = discover_samples()
    if not samples:
        raise SystemExit(
            f"No images under {config.RAW_DIR}. Add a dataset or run "
            f"'python -m scripts.make_synthetic_data'."
        )
    train, val, test = stratified_split(samples)
    save_splits(train, val, test)
    return train, val, test


def train_cnn(train, val, epochs, seed=config.RANDOM_SEED):
    import tensorflow as tf

    from .model_cnn import build_cnn
    from .preprocessing import make_tf_dataset

    tf.keras.utils.set_random_seed(seed)
    model = build_cnn()
    train_ds = make_tf_dataset(train, training=True)
    val_ds = make_tf_dataset(val, training=False)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5,
        ),
    ]
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=compute_class_weights(train), callbacks=callbacks, verbose=2,
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.CNN_MODEL_FILE)
    _save_history("cnn", hist, epochs)
    _tune_dl_threshold("cnn", model, val)
    print(f"[cnn] saved -> {config.CNN_MODEL_FILE}")
    return model


def train_transfer(train, val, epochs, seed=config.RANDOM_SEED):
    import tensorflow as tf

    from .model_transfer import build_transfer
    from .preprocessing import make_tf_dataset

    tf.keras.utils.set_random_seed(seed)
    model = build_transfer()
    train_ds = make_tf_dataset(train, training=True)
    val_ds = make_tf_dataset(val, training=False)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True,
        )
    ]
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=compute_class_weights(train), callbacks=callbacks, verbose=2,
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.TRANSFER_MODEL_FILE)
    _save_history("transfer", hist, epochs)
    _tune_dl_threshold("transfer", model, val)
    print(f"[transfer] saved -> {config.TRANSFER_MODEL_FILE}")
    return model


def train_classical(train, val):
    """Fit the classical baseline on the TRAIN split only.

    It previously fitted on train+val to use more data, but that silently broke
    threshold selection: ``scripts.threshold_analysis`` picks each model's
    operating point on the validation split, and if the classical model has
    already seen those images its validation scores are optimistic. The
    consequence was concrete - a false-positive-rate ceiling looked satisfied on
    val while the real test FPR was 66%, so normal X-rays were reported as
    fractured. Training on train only keeps val a genuine holdout and puts the
    classical model on the same footing as the deep models.
    """
    import numpy as np
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from .features_classical import build_feature_matrix
    from .model_svm import build_classical, save_classical

    fit_samples = train
    X, y = build_feature_matrix(fit_samples)
    print(f"[classical] feature matrix: {X.shape}")

    pipeline = build_classical()

    # Small, sensible grid; guarded so it still works on tiny datasets.
    n_per_class = np.bincount(y).min() if len(set(y)) > 1 else len(y)
    n_splits = int(max(2, min(3, n_per_class)))
    if config.CLASSICAL_MODEL_TYPE.lower() == "svm":
        grid = {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]}
    else:
        grid = {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 20]}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X, y)
    print(f"[classical] best params: {search.best_params_}  "
          f"cv ROC-AUC={search.best_score_:.3f}")

    # Tune the decision threshold on the held-out VALIDATION split - the same
    # data the deep models use - so every model's operating point is chosen the
    # same way and on images none of them trained on.
    from .model_svm import predict_proba_positive
    from .thresholds import best_threshold_f1, save_threshold

    best = search.best_estimator_
    if val:
        Xv, yv = build_feature_matrix(val)
        pv = predict_proba_positive(best, Xv)
        t = best_threshold_f1(yv, pv)
        print(f"[classical] tuned decision threshold on val -> {t:.3f}")
    else:
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_predict

        oof = cross_val_predict(clone(best), X, y, cv=cv, method="predict_proba")
        pos_col = config.POSITIVE_INDEX if oof.shape[1] > 1 else 0
        t = best_threshold_f1(y, oof[:, pos_col])
        print(f"[classical] tuned decision threshold (oof fallback) -> {t:.3f}")
    save_threshold("classical", t)

    save_classical(best)
    print(f"[classical] saved -> {config.CLASSICAL_MODEL_FILE}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RadFrac models")
    parser.add_argument("--model", choices=["all", "cnn", "transfer", "classical"],
                        default="all")
    parser.add_argument("--epochs_cnn", type=int, default=config.EPOCHS_CNN)
    parser.add_argument("--epochs_transfer", type=int, default=config.EPOCHS_TRANSFER)
    args = parser.parse_args()

    config.ensure_output_dirs()
    train, val, test = _ensure_splits()
    print(f"Using splits: train={len(train)} val={len(val)} test={len(test)}")

    start = time.time()
    trained: list[str] = []

    if args.model in ("all", "cnn"):
        train_cnn(train, val, args.epochs_cnn)
        trained.append("cnn")
    if args.model in ("all", "transfer"):
        train_transfer(train, val, args.epochs_transfer)
        trained.append("transfer")
    if args.model in ("all", "classical"):
        train_classical(train, val)
        trained.append("classical")

    elapsed = time.time() - start
    print(f"\nTrained {trained} in {elapsed:.1f}s")
    # Record what was trained for provenance.
    (config.METRICS_DIR / "train_manifest.json").write_text(
        json.dumps({"trained": trained, "elapsed_sec": round(elapsed, 1),
                    "classical_type": config.CLASSICAL_MODEL_TYPE}, indent=2)
    )


if __name__ == "__main__":
    main()
