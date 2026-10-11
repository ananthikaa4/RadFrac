"""Fit the two-model ensemble on VALIDATION, freeze it, then score TEST once.

Compares, on validation:
  A  SVM alone
  B  MobileNetV2 alone
  C  hard voting - reported as AND and OR, since majority is undefined for two
  D  equal-weight probability average (after calibration)
  E  weight-optimised average (after calibration)

The winner by validation ROC-AUC becomes the frozen ensemble. Only then is the
clean test split scored, so no test information influences any choice.

Usage::

    python -m scripts.fit_ensemble
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.ensemble import (  # noqa: E402
    MEMBER_LABELS,
    MEMBERS,
    EnsembleConfig,
    PlattCalibrator,
    score,
    score_hard,
)
from src.preprocessing import load_splits, splits_exist  # noqa: E402
from src.threshold_analysis import sweep_costs  # noqa: E402


def member_probs(key: str, samples) -> np.ndarray:
    """P(fractured) for a split from one member model."""
    if key == "transfer":
        import tensorflow as tf
        from src.preprocessing import make_tf_dataset

        m = tf.keras.models.load_model(config.TRANSFER_MODEL_FILE)
        ds = make_tf_dataset(samples, training=False, shuffle=False)
        return m.predict(ds, verbose=0).reshape(-1)

    from src.features_classical import build_feature_matrix
    from src.model_svm import load_classical, predict_proba_positive

    X, _ = build_feature_matrix(samples)
    return predict_proba_positive(load_classical(), X)


def _row(name, m):
    auc = "  n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
    return (f"{name:<34}{m['accuracy']:>9.4f}{m['precision']:>10.4f}"
            f"{m['recall']:>8.4f}{m['f1']:>8.4f}{auc:>9}"
            f"{m['fn']:>6}{m['fp']:>6}")


HEADER = (f"{'configuration':<34}{'acc':>9}{'prec':>10}{'rec':>8}"
          f"{'f1':>8}{'roc_auc':>9}{'FN':>6}{'FP':>6}")


def main() -> None:
    if not splits_exist():
        raise SystemExit("No splits. Set RADFRAC_SPLIT_DIR / run src.preprocessing.")
    for key, path in (("classical", config.CLASSICAL_MODEL_FILE),
                      ("transfer", config.TRANSFER_MODEL_FILE)):
        if not path.exists():
            raise SystemExit(f"missing artifact for {key}: {path}")

    train, val, test = load_splits()
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")
    print(f"split dir: {config.SPLIT_DIR}\n")

    y_val = np.array([s.label for s in val])
    y_test = np.array([s.label for s in test])

    print("scoring members ...")
    raw_val, raw_test = {}, {}
    for m in MEMBERS:
        raw_val[m] = member_probs(m, val)
        raw_test[m] = member_probs(m, test)
        print(f"    {m:<10} val={len(raw_val[m])} test={len(raw_test[m])}")

    # ---------------- calibrate on VALIDATION -----------------------------
    print("\ncalibrating members on validation (Platt) ...")
    cals: dict[str, PlattCalibrator] = {}
    cal_val, cal_test = {}, {}
    for m in MEMBERS:
        c = PlattCalibrator().fit(raw_val[m], y_val)
        cals[m] = c
        cal_val[m] = c.transform(raw_val[m])
        cal_test[m] = c.transform(raw_test[m])
        from src.threshold_analysis import calibration
        before = calibration(y_val, raw_val[m]).ece
        after = calibration(y_val, cal_val[m]).ece
        print(f"    {m:<10} val ECE {before:.4f} -> {after:.4f}  "
              f"(coef={c.coef:.3f}, intercept={c.intercept:.3f})")

    # ---------------- per-member thresholds on VALIDATION -----------------
    member_thr = {}
    for m in MEMBERS:
        a = sweep_costs(y_val, cal_val[m], recall_target=None)
        member_thr[m] = a.best_threshold

    # ---------------- candidate configurations, judged on VAL ------------
    print("\n" + "=" * 90)
    print("VALIDATION comparison (all selection happens here)")
    print("=" * 90)
    print(HEADER)
    print("-" * 90)

    val_results = {}
    for m in MEMBERS:
        val_results[m] = score(y_val, cal_val[m], member_thr[m])
        print(_row(f"{MEMBER_LABELS[m]}", val_results[m]))

    hard_preds_val = {m: (cal_val[m] >= member_thr[m]).astype(int) for m in MEMBERS}
    val_and = score_hard(y_val, hard_preds_val, "and")
    val_or = score_hard(y_val, hard_preds_val, "or")
    print(_row("hard vote AND (both positive)", val_and))
    print(_row("hard vote OR (either positive)", val_or))

    # D: equal average
    avg_val = 0.5 * cal_val["classical"] + 0.5 * cal_val["transfer"]
    a_avg = sweep_costs(y_val, avg_val, recall_target=None)
    val_avg = score(y_val, avg_val, a_avg.best_threshold)
    print(_row("equal average (0.50 / 0.50)", val_avg))

    # E: weight search on VAL by ROC-AUC (threshold-independent)
    from sklearn.metrics import roc_auc_score

    best_w, best_auc = 0.5, -1.0
    for w in np.arange(0.0, 1.0001, 0.05):
        blend = w * cal_val["classical"] + (1 - w) * cal_val["transfer"]
        auc = roc_auc_score(y_val, blend)
        if auc > best_auc:
            best_auc, best_w = float(auc), float(w)
    wt_val_probs = best_w * cal_val["classical"] + (1 - best_w) * cal_val["transfer"]
    a_wt = sweep_costs(y_val, wt_val_probs, recall_target=None)
    val_wt = score(y_val, wt_val_probs, a_wt.best_threshold)
    print(_row(f"weighted ({best_w:.2f} svm / {1 - best_w:.2f} mobile)", val_wt))
    print("-" * 90)

    # ---------------- choose + FREEZE -------------------------------------
    soft_candidates = {
        "equal-average": (val_avg, 0.5, a_avg.best_threshold),
        "weighted-average": (val_wt, best_w, a_wt.best_threshold),
    }
    chosen_name = max(soft_candidates,
                      key=lambda k: soft_candidates[k][0]["roc_auc"] or 0.0)
    chosen_metrics, chosen_w, chosen_thr = soft_candidates[chosen_name]

    cfg = EnsembleConfig(
        members=list(MEMBERS),
        weights={"classical": round(chosen_w, 4),
                 "transfer": round(1 - chosen_w, 4)},
        calibrators={m: cals[m].to_dict() for m in MEMBERS},
        threshold=float(chosen_thr),
        threshold_source=(f"validation cost-constrained sweep "
                          f"(FN x{config.FALSE_NEGATIVE_COST:g}, "
                          f"FPR <= {config.MAX_FALSE_POSITIVE_RATE:.2f})"),
        method=chosen_name,
        val_metrics={
            "chosen": chosen_metrics,
            "members": {m: val_results[m] for m in MEMBERS},
            "hard_and": val_and, "hard_or": val_or,
            "equal_average": val_avg, "weighted_average": val_wt,
            "member_thresholds": member_thr,
        },
        notes=("Platt calibration, blend weights and threshold all fitted on the "
               "validation split only. Majority voting is undefined for two "
               "members, so hard voting is reported as AND and OR."),
    )
    cfg.save()
    print(f"\nFROZEN -> {config.ENSEMBLE_CONFIG_FILE}")
    print(f"  method    : {cfg.method}")
    print(f"  weights   : svm={cfg.weights['classical']:.2f} "
          f"mobilenet={cfg.weights['transfer']:.2f}")
    print(f"  threshold : {cfg.threshold:.4f}")

    # ---------------- single TEST evaluation ------------------------------
    print("\n" + "=" * 90)
    print("TEST evaluation at the frozen configuration (scored once)")
    print("=" * 90)
    print(HEADER)
    print("-" * 90)

    test_results = {}
    for m in MEMBERS:
        test_results[m] = score(y_test, cal_test[m], member_thr[m])
        print(_row(MEMBER_LABELS[m], test_results[m]))

    ens_test_p = cfg.combine({m: cal_test[m] for m in MEMBERS})
    ens_test = score(y_test, ens_test_p, cfg.threshold)
    print(_row("RADFRAC Ensemble (2 models)", ens_test))

    hard_preds_test = {m: (cal_test[m] >= member_thr[m]).astype(int) for m in MEMBERS}
    t_and = score_hard(y_test, hard_preds_test, "and")
    t_or = score_hard(y_test, hard_preds_test, "or")
    print(_row("  [ref] hard vote AND", t_and))
    print(_row("  [ref] hard vote OR", t_or))
    print("-" * 90)

    # Honest verdict
    best_member = max(MEMBERS, key=lambda m: test_results[m]["roc_auc"] or 0)
    bm_auc = test_results[best_member]["roc_auc"]
    e_auc = ens_test["roc_auc"]
    print(f"\nbest single member on test: {MEMBER_LABELS[best_member]} "
          f"(ROC-AUC {bm_auc:.4f})")
    print(f"ensemble on test          : ROC-AUC {e_auc:.4f}")
    delta = e_auc - bm_auc
    if delta > 0.005:
        verdict = f"ensemble IMPROVES on the best single model (+{delta:.4f} AUC)"
    elif delta < -0.005:
        verdict = f"ensemble is WORSE than the best single model ({delta:+.4f} AUC)"
    else:
        verdict = f"ensemble is EQUIVALENT to the best single model ({delta:+.4f} AUC)"
    print(f"verdict: {verdict}")

    payload = {
        "split_dir": str(config.SPLIT_DIR),
        "n_val": len(val), "n_test": len(test),
        "frozen_config": cfg.to_dict(),
        "test": {
            "members": {m: test_results[m] for m in MEMBERS},
            "ensemble": ens_test,
            "hard_and": t_and, "hard_or": t_or,
        },
        "verdict": verdict,
    }
    out = config.METRICS_DIR / "ensemble_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
