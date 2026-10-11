"""Run cost-sensitive threshold selection and calibration analysis.

Methodology (no leakage):
  * Operating points are chosen on the **validation** split.
  * The **test** split is then scored once, at those fixed thresholds, purely
    for reporting. Nothing is tuned on test data.

Outputs
  outputs/models/cost_thresholds.json    cost-optimal threshold per model
  outputs/metrics/threshold_analysis.json full sweep + chosen points
  outputs/metrics/calibration.json        ECE / Brier / reliability bins
  outputs/figures/cost_threshold.png      cost curve + PR trade-off
  outputs/figures/calibration.png         reliability curves

Usage::

    python -m scripts.threshold_analysis
    python -m scripts.threshold_analysis --fn-cost 10 --recall-target 0.98
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dataset_status import banner, is_synthetic_data  # noqa: E402
from src.preprocessing import load_splits, splits_exist  # noqa: E402
from src.threshold_analysis import (  # noqa: E402
    calibration,
    plot_calibration,
    plot_cost_curve,
    save_cost_thresholds,
    sweep_costs,
)


def _probs_for(model_key: str, samples) -> np.ndarray:
    """Predicted P(fractured) for a split, per model family."""
    if model_key in ("cnn", "transfer"):
        import tensorflow as tf
        from src.preprocessing import make_tf_dataset

        path = (config.CNN_MODEL_FILE if model_key == "cnn"
                else config.TRANSFER_MODEL_FILE)
        model = tf.keras.models.load_model(path)
        ds = make_tf_dataset(samples, training=False, shuffle=False)
        return model.predict(ds, verbose=0).reshape(-1)

    from src.features_classical import build_feature_matrix
    from src.model_svm import load_classical, predict_proba_positive

    pipe = load_classical()
    X, _ = build_feature_matrix(samples)
    return predict_proba_positive(pipe, X)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cost-sensitive threshold + calibration")
    ap.add_argument("--fn-cost", type=float, default=config.FALSE_NEGATIVE_COST)
    ap.add_argument("--fp-cost", type=float, default=config.FALSE_POSITIVE_COST)
    ap.add_argument("--recall-target", type=float, default=0.95)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    config.ensure_output_dirs()
    if not splits_exist():
        raise SystemExit("No splits found. Run 'python -m src.preprocessing' first.")
    _, val, test = load_splits()
    if not val or not test:
        raise SystemExit("Validation/test split is empty.")

    y_val = np.array([s.label for s in val])
    y_test = np.array([s.label for s in test])

    artifacts = {
        "cnn": config.CNN_MODEL_FILE,
        "transfer": config.TRANSFER_MODEL_FILE,
        "classical": config.CLASSICAL_MODEL_FILE,
    }

    cost_analyses, calib_results = {}, {}
    chosen: dict[str, float] = {}
    report: dict[str, dict] = {}

    for key, artifact in artifacts.items():
        if not artifact.exists():
            print(f"[{key}] artifact missing; skipping.")
            continue

        print(f"[{key}] scoring validation ({len(val)}) and test ({len(test)}) ...")
        p_val = _probs_for(key, val)
        p_test = _probs_for(key, test)

        # --- choose operating point on VALIDATION only -----------------------
        analysis = sweep_costs(y_val, p_val, fn_cost=args.fn_cost,
                               fp_cost=args.fp_cost,
                               recall_target=args.recall_target)
        cost_analyses[key] = analysis
        chosen[key] = analysis.best_threshold

        # --- report that fixed point on TEST --------------------------------
        t = analysis.best_threshold
        pred_test = (p_test >= t).astype(int)
        tp = int(np.sum((y_test == 1) & (pred_test == 1)))
        fp = int(np.sum((y_test == 0) & (pred_test == 1)))
        fn = int(np.sum((y_test == 1) & (pred_test == 0)))
        tn = int(np.sum((y_test == 0) & (pred_test == 0)))
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0

        calib = calibration(y_test, p_test, n_bins=args.bins)
        calib_results[key] = calib

        report[key] = {
            "cost_optimal_threshold_from_val": t,
            "val_cost_at_threshold": analysis.best_cost,
            "val_stats_at_threshold": analysis.at_best,
            "threshold_for_recall_target": analysis.threshold_for_recall,
            "recall_target": args.recall_target,
            "test_at_cost_threshold": {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": round(recall, 4), "precision": round(precision, 4),
                "false_negatives": fn,
            },
            "calibration": calib.to_dict(),
        }

        print(f"    cost-optimal t={t:.3f} (val)  ->  test: FN={fn} FP={fp} "
              f"recall={recall:.3f} precision={precision:.3f}")
        print(f"    calibration: ECE={calib.ece:.3f} Brier={calib.brier:.3f} "
              f"- {calib.verdict}")

    if not report:
        raise SystemExit("No trained models found. Run 'python -m src.train' first.")

    save_cost_thresholds(chosen)
    payload = {
        "fn_cost": args.fn_cost, "fp_cost": args.fp_cost,
        "recall_target": args.recall_target,
        "methodology": ("thresholds selected on the validation split; test split "
                        "scored once at those fixed thresholds"),
        "synthetic_data": bool(is_synthetic_data()),
        "models": report,
    }
    config.THRESHOLD_ANALYSIS_FILE.write_text(json.dumps(payload, indent=2))
    config.CALIBRATION_FILE.write_text(json.dumps(
        {k: v.to_dict() for k, v in calib_results.items()}, indent=2))

    plot_cost_curve(cost_analyses, config.FIGURES_DIR / "cost_threshold.png")
    plot_calibration(calib_results, config.FIGURES_DIR / "calibration.png")

    if is_synthetic_data():
        print(banner())

    print("\nFalse negatives are the clinically costly error at a screening step;")
    print(f"cost weighting used: FN x {args.fn_cost:g} vs FP x {args.fp_cost:g}.")
    print(f"\ncost thresholds -> {config.COST_THRESHOLDS_FILE}")
    print(f"analysis        -> {config.THRESHOLD_ANALYSIS_FILE}")
    print(f"calibration     -> {config.CALIBRATION_FILE}")
    print(f"figures         -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
