"""Train the gradient-boosting models (XGBoost, LightGBM, CatBoost).

Pipeline::

    split manifests -> hybrid features (cached) -> for each booster:
        Optuna TPE search over {PCA k, booster hyperparameters}
            objective = ROC-AUC on the VALIDATION split (early stopping on val)
            every trial -> MLflow nested run + CSV row
        refit best config on TRAIN
        cost-constrained threshold on VALIDATION (FN x5, FPR <= 0.25)
        score the TEST split exactly once
    -> boosting committee (mean of the three) thresholded on validation
    -> outputs/models/boosting/{key}.joblib + registry.json
    -> outputs/metrics/boosting_predictions.npz  (val/test scores for analysis)

Usage (inside the Docker image, because the deep features need TensorFlow)::

    python -m scripts.train_boosting                 # all three
    python -m scripts.train_boosting --models xgboost --trials 10
    python -m scripts.train_boosting --features-only # just build the cache

Use the leakage-free group-aware split::

    RADFRAC_SPLIT_DIR=../data/splits_grouped python -m scripts.train_boosting
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.ensemble import score  # noqa: E402
from src.features_hybrid import assemble, extract_split  # noqa: E402
from src.model_boosting import (  # noqa: E402
    BOOSTER_KEYS, COMMITTEE_KEY, DISPLAY_NAMES, assemble_pipeline, build_classifier,
    build_preprocessor, load_yaml_config, positive_proba, save_model, save_registry,
)
from src.preprocessing import load_splits  # noqa: E402
from src.threshold_analysis import sweep_costs  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features(cfg: dict, force: bool = False) -> dict[str, dict]:
    cache_dir = config.PROJECT_ROOT / cfg["features"]["cache_dir"]
    train, val, test = load_splits()
    print(f"split dir: {config.SPLIT_DIR}  train={len(train)} val={len(val)} test={len(test)}")
    transfer = None
    if force or not all((cache_dir / f"hybrid_{s}.npz").exists() for s in ("train", "val", "test")):
        from src.features_hybrid import load_transfer_model
        transfer = load_transfer_model()
    out = {}
    for name, samples in (("train", train), ("val", val), ("test", test)):
        t0 = time.time()
        out[name] = extract_split(samples, name, cache_dir, transfer_model=transfer, force=force)
        print(f"[features] {name}: handcrafted {out[name]['handcrafted'].shape} "
              f"deep {out[name]['deep'].shape} ({time.time() - t0:.1f}s)")
    return out


# --------------------------------------------------------------------------- #
# Optuna helpers
# --------------------------------------------------------------------------- #
def suggest(trial, name: str, spec: dict):
    low, high, log = spec["low"], spec["high"], bool(spec.get("log", False))
    if isinstance(low, int) and isinstance(high, int) and not log:
        return trial.suggest_int(name, low, high)
    return trial.suggest_float(name, float(low), float(high), log=log)


def fit_with_early_stopping(key: str, params: dict, max_est: int, es_rounds: int,
                            Xtr, ytr, Xv, yv, seed: int):
    """Fit with early stopping on val; return (model, best_iteration)."""
    if key == "xgboost":
        clf = build_classifier(key, {**params, "early_stopping_rounds": es_rounds},
                               max_est, seed)
        clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        return clf, int(clf.best_iteration) + 1
    if key == "lightgbm":
        import lightgbm as lgb
        clf = build_classifier(key, params, max_est, seed)
        clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        return clf, int(clf.best_iteration_ or max_est)
    if key == "catboost":
        clf = build_classifier(key, params, max_est, seed)
        clf.fit(Xtr, ytr, eval_set=(Xv, yv), early_stopping_rounds=es_rounds)
        return clf, int(clf.get_best_iteration()) + 1
    raise ValueError(key)


def learning_curves(key: str, clf) -> dict:
    """Train/val log-loss per boosting round from the final fit."""
    if key == "xgboost":
        r = clf.evals_result()
        return {"train": r["validation_0"]["logloss"], "val": r["validation_1"]["logloss"]}
    if key == "lightgbm":
        r = clf.evals_result_
        names = list(r)
        return {"train": r[names[0]]["binary_logloss"], "val": r[names[1]]["binary_logloss"]}
    if key == "catboost":
        r = clf.get_evals_result()
        names = [n for n in r if n != "learn"]
        return {"train": r["learn"]["Logloss"], "val": r[names[-1]]["Logloss"]}
    return {}


def fit_final(key: str, params: dict, n_est: int, Xtr, ytr, Xv, yv, seed: int):
    clf = build_classifier(key, params, n_est, seed)
    if key == "xgboost":
        clf.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xv, yv)], verbose=False)
    elif key == "lightgbm":
        clf.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xv, yv)], eval_metric="binary_logloss")
    else:
        clf.fit(Xtr, ytr, eval_set=(Xv, yv), use_best_model=False)
    return clf


# --------------------------------------------------------------------------- #
# Tuning one booster
# --------------------------------------------------------------------------- #
def tune_booster(key: str, cfg: dict, X: dict, y: dict, n_trials: int, mlflow) -> dict:
    import optuna
    from sklearn.metrics import roc_auc_score

    seed = cfg["seed"]
    space = cfg["search_spaces"][key]
    pca_choices = cfg["search_spaces"]["common"]["pca_components"]
    es = cfg["tuning"]["early_stopping_rounds"]
    max_est = int(space["max_estimators"])
    whiten = bool(cfg["dimensionality_reduction"].get("whiten", False))

    # PCA is deterministic per k, so fit each projection once and reuse it.
    proj_cache: dict[int, tuple] = {}

    def projected(k: int):
        if k not in proj_cache:
            pre = build_preprocessor(k, seed, whiten).fit(X["train"])
            proj_cache[k] = (pre, pre.transform(X["train"]), pre.transform(X["val"]))
        return proj_cache[k]

    log_dir = config.PROJECT_ROOT / cfg["tracking"]["csv_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"trials_{key}.csv"
    rows: list[dict] = []

    def objective(trial):
        k = trial.suggest_categorical("pca_components", pca_choices)
        params = {n: suggest(trial, n, s) for n, s in space.items() if n != "max_estimators"}
        _, Ztr, Zv = projected(k)
        t0 = time.time()
        clf, best_it = fit_with_early_stopping(key, params, max_est, es,
                                               Ztr, y["train"], Zv, y["val"], seed)
        auc = float(roc_auc_score(y["val"], positive_proba(clf, Zv)))
        trial.set_user_attr("best_iteration", best_it)
        row = {"trial": trial.number, "val_roc_auc": auc, "best_iteration": best_it,
               "fit_seconds": round(time.time() - t0, 2), "pca_components": k, **params}
        rows.append(row)
        with mlflow.start_run(run_name=f"{key}-trial-{trial.number:03d}", nested=True):
            mlflow.log_params({"pca_components": k, **params})
            mlflow.log_metrics({"val_roc_auc": auc, "best_iteration": best_it,
                                "fit_seconds": row["fit_seconds"]})
        print(f"  [{key}] trial {trial.number:02d}  k={k:<3d}  val AUC={auc:.4f}  "
              f"iters={best_it}", flush=True)
        return auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"radfrac-{key}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials,
                   timeout=cfg["tuning"].get("timeout_sec_per_model"))

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = study.best_trial
    best_params = {k: v for k, v in best.params.items() if k != "pca_components"}
    k = int(best.params["pca_components"])
    n_est = int(best.user_attrs["best_iteration"])

    # Param importance (fANOVA) - informative for the report; optional.
    try:
        importance = {p: float(v) for p, v in
                      optuna.importance.get_param_importances(study).items()}
    except Exception:
        importance = {}

    pre, Ztr, Zv = projected(k)
    clf = fit_final(key, best_params, n_est, Ztr, y["train"], Zv, y["val"], seed)
    pipe = assemble_pipeline(pre, clf)
    return {
        "pipeline": pipe, "best_params": best_params, "pca_components": k,
        "n_estimators": n_est, "best_val_auc_search": float(best.value),
        "n_trials": len(study.trials), "param_importance": importance,
        "curves": learning_curves(key, clf),
        "explained_variance": float(pre.named_steps["pca"].explained_variance_ratio_.sum()),
        "trials_csv": str(csv_path.relative_to(config.PROJECT_ROOT)),
    }


def operating_point(cfg: dict, yv: np.ndarray, pv: np.ndarray):
    op = cfg["operating_point"]
    return sweep_costs(yv, pv, fn_cost=op["fn_cost"], fp_cost=op["fp_cost"],
                       max_fpr=op["max_fpr"], recall_target=0.95)


def _clean(m: dict) -> dict:
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in m.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Tune + train gradient-boosting models")
    ap.add_argument("--models", nargs="+", choices=list(BOOSTER_KEYS), default=list(BOOSTER_KEYS))
    ap.add_argument("--trials", type=int, default=None, help="override trials per model")
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    np.random.seed(cfg["seed"])
    feats = load_features(cfg, force=args.force_features)
    if args.features_only:
        return

    fc = cfg["features"]
    X = {s: assemble(feats[s], fc["use_handcrafted"], fc["use_deep"]) for s in feats}
    y = {s: feats[s]["y"] for s in feats}
    print(f"hybrid feature dim = {X['train'].shape[1]}")

    import mlflow
    uri_dir = (config.PROJECT_ROOT / cfg["tracking"]["mlflow_uri"]).resolve()
    uri_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri_dir.as_uri())
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    from src.model_boosting import load_registry
    registry = load_registry() or {}
    registry.setdefault("models", {})
    registry["features"] = {
        "use_handcrafted": fc["use_handcrafted"], "use_deep": fc["use_deep"],
        "dim": int(X["train"].shape[1]),
        "description": "HOG(1764)+Canny grid(16) + MobileNetV2 GAP(1280) -> StandardScaler -> PCA",
    }
    registry["split_dir"] = str(config.SPLIT_DIR)
    registry["n"] = {s: int(len(y[s])) for s in y}

    preds = {"y_val": y["val"], "y_test": y["test"],
             "paths_test": feats["test"]["paths"], "paths_val": feats["val"]["paths"],
             "transfer_val": feats["val"]["transfer_prob"],
             "transfer_test": feats["test"]["transfer_prob"]}
    old_preds = config.METRICS_DIR / "boosting_predictions.npz"
    if old_preds.exists():   # keep scores of boosters not retrained this run
        d = np.load(old_preds, allow_pickle=False)
        for k in d.files:
            preds.setdefault(k, d[k])

    n_trials_cfg = cfg["tuning"]["n_trials"]
    for key in args.models:
        n_trials = args.trials or int(n_trials_cfg[key])
        print(f"\n=== {DISPLAY_NAMES[key]}: {n_trials} Optuna trials ===")
        t0 = time.time()
        with mlflow.start_run(run_name=f"{key}-search") as run:
            mlflow.log_params({"model": key, "n_trials": n_trials,
                               "feature_dim": X["train"].shape[1],
                               "split_dir": str(config.SPLIT_DIR)})
            res = tune_booster(key, cfg, X, y, n_trials, mlflow)
            pipe = res["pipeline"]
            pv = positive_proba(pipe, X["val"])
            pt = positive_proba(pipe, X["test"])
            ca = operating_point(cfg, y["val"], pv)
            thr = float(ca.best_threshold)
            val_m = _clean(score(y["val"], pv, thr))
            test_m = _clean(score(y["test"], pt, thr))
            mlflow.log_params({f"best_{k}": v for k, v in res["best_params"].items()})
            mlflow.log_params({"best_pca_components": res["pca_components"],
                               "best_n_estimators": res["n_estimators"],
                               "threshold": round(thr, 4)})
            mlflow.log_metrics({"val_roc_auc": val_m["roc_auc"], "val_f1": val_m["f1"],
                                "test_roc_auc": test_m["roc_auc"], "test_f1": test_m["f1"],
                                "test_accuracy": test_m["accuracy"],
                                "test_recall": test_m["recall"],
                                "test_precision": test_m["precision"],
                                "pca_explained_variance": res["explained_variance"]})
            path = save_model(key, pipe)
            mlflow.log_artifact(str(path))
            run_id = run.info.run_id

        registry["models"][key] = {
            "display_name": DISPLAY_NAMES[key],
            "artifact": str(path.relative_to(config.PROJECT_ROOT)),
            "best_params": res["best_params"],
            "pca_components": res["pca_components"],
            "pca_explained_variance": res["explained_variance"],
            "n_estimators": res["n_estimators"],
            "n_trials": res["n_trials"],
            "best_val_auc_search": res["best_val_auc_search"],
            "param_importance": res["param_importance"],
            "threshold": thr,
            "threshold_source": f"validation cost-constrained sweep (FN x{ca.fn_cost:g}, "
                                f"FPR <= {ca.max_fpr:.2f})",
            "val_metrics": val_m,
            "test_metrics": test_m,
            "learning_curves": res["curves"],
            "trials_csv": res["trials_csv"],
            "mlflow_run_id": run_id,
            "train_seconds": round(time.time() - t0, 1),
        }
        preds[f"{key}_val"], preds[f"{key}_test"] = pv, pt
        print(f"[{key}] val AUC={val_m['roc_auc']:.4f}  TEST AUC={test_m['roc_auc']:.4f}  "
              f"F1={test_m['f1']:.4f}  thr={thr:.3f}  ({time.time() - t0:.0f}s)")
        save_registry(registry)

    # ---- committee: soft vote of the three boosters ------------------------ #
    if all(f"{k}_val" in preds for k in BOOSTER_KEYS):
        cv = np.mean([preds[f"{k}_val"] for k in BOOSTER_KEYS], axis=0)
        ct = np.mean([preds[f"{k}_test"] for k in BOOSTER_KEYS], axis=0)
        ca = operating_point(cfg, y["val"], cv)
        thr = float(ca.best_threshold)
        registry["models"][COMMITTEE_KEY] = {
            "display_name": DISPLAY_NAMES[COMMITTEE_KEY],
            "members": list(BOOSTER_KEYS), "method": "mean probability (soft vote)",
            "threshold": thr,
            "threshold_source": "validation cost-constrained sweep",
            "val_metrics": _clean(score(y["val"], cv, thr)),
            "test_metrics": _clean(score(y["test"], ct, thr)),
        }
        preds[f"{COMMITTEE_KEY}_val"], preds[f"{COMMITTEE_KEY}_test"] = cv, ct
        print(f"[committee] TEST AUC={registry['models'][COMMITTEE_KEY]['test_metrics']['roc_auc']:.4f}")

    save_registry(registry)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.METRICS_DIR / "boosting_predictions.npz", **preds)
    (config.METRICS_DIR / "boosting_results.json").write_text(json.dumps(
        {k: {"val": v.get("val_metrics"), "test": v.get("test_metrics"),
             "threshold": v.get("threshold")} for k, v in registry["models"].items()},
        indent=2))
    print("\nregistry ->", config.MODELS_DIR / "boosting" / "registry.json")


if __name__ == "__main__":
    main()
