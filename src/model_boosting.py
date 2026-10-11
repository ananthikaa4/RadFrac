"""Gradient-boosting models: XGBoost, LightGBM and CatBoost (+ committee).

Each model is a scikit-learn ``Pipeline``::

    hybrid features (3,060-d) -> StandardScaler -> PCA(k) -> booster

so scaling and the PCA projection are fitted on the training split only and
travel with the model. ``k`` and every booster hyperparameter are chosen by
Optuna on the validation split (``scripts/train_boosting.py``).

These models are *additional* to the existing classical SVM, MobileNetV2 and
their ensemble; they do not change any existing artifact, threshold or the
default ensemble verdict. They are persisted under ``outputs/models/boosting``
with a ``registry.json`` that records their thresholds and metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

BOOSTER_KEYS = ("xgboost", "lightgbm", "catboost")
COMMITTEE_KEY = "boost_committee"
ADVANCED_KEYS = BOOSTER_KEYS + (COMMITTEE_KEY,)

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    COMMITTEE_KEY: "Boosting Committee",
}
DESCRIPTIONS = {
    "xgboost": "Level-wise gradient-boosted trees",
    "lightgbm": "Leaf-wise histogram boosting",
    "catboost": "Ordered boosting, symmetric trees",
    COMMITTEE_KEY: "Soft vote of the three boosters",
}

CONFIG_FILE = config.PROJECT_ROOT / "configs" / "boosting.yaml"


def boosting_dir() -> Path:
    # Resolved at call time so tests that repoint config.MODELS_DIR take effect.
    return Path(config.MODELS_DIR) / "boosting"


def registry_file() -> Path:
    return boosting_dir() / "registry.json"


# --------------------------------------------------------------------------- #
# YAML configuration
# --------------------------------------------------------------------------- #
def load_yaml_config(path: Path | str | None = None) -> dict:
    import yaml

    path = Path(path) if path else CONFIG_FILE
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_classifier(key: str, params: dict, n_estimators: int, seed: int = config.RANDOM_SEED):
    """Unfitted booster with the given hyperparameters."""
    if key == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_estimators, tree_method="hist", eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0, **params,
        )
    if key == "lightgbm":
        from lightgbm import LGBMClassifier
        p = dict(params)
        # sklearn-API names for the LightGBM aliases used in the YAML
        mapping = {"feature_fraction": "colsample_bytree",
                   "bagging_fraction": "subsample",
                   "lambda_l1": "reg_alpha", "lambda_l2": "reg_lambda"}
        p = {mapping.get(k, k): v for k, v in p.items()}
        return LGBMClassifier(
            n_estimators=n_estimators, subsample_freq=1, random_state=seed,
            n_jobs=-1, verbose=-1, **p,
        )
    if key == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=n_estimators, random_seed=seed, verbose=0,
            thread_count=-1, allow_writing_files=False, **params,
        )
    raise ValueError(f"unknown booster {key!r}")


def build_preprocessor(n_components: int, seed: int = config.RANDOM_SEED, whiten: bool = False):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=whiten,
                    svd_solver="randomized", random_state=seed)),
    ])


def assemble_pipeline(preprocessor, classifier):
    """Join an already-fitted preprocessor and classifier into one Pipeline."""
    from sklearn.pipeline import Pipeline

    return Pipeline([("scaler", preprocessor.named_steps["scaler"]),
                     ("pca", preprocessor.named_steps["pca"]),
                     ("clf", classifier)])


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(config.POSITIVE_INDEX)]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def model_path(key: str) -> Path:
    return boosting_dir() / f"{key}.joblib"


def save_model(key: str, pipeline) -> Path:
    import joblib

    path = model_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(key: str):
    import joblib

    return joblib.load(model_path(key))


def load_registry() -> dict:
    f = registry_file()
    if f.exists():
        return json.loads(f.read_text())
    return {}


def save_registry(reg: dict) -> None:
    f = registry_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(reg, indent=2))


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
@dataclass
class BoostingSuite:
    """Loaded boosters + their frozen thresholds, used by the service layer."""
    models: dict = field(default_factory=dict)          # key -> Pipeline
    thresholds: dict = field(default_factory=dict)      # key -> float
    test_metrics: dict = field(default_factory=dict)    # key -> dict
    features: dict = field(default_factory=dict)        # feature spec from registry

    @classmethod
    def load(cls) -> "BoostingSuite | None":
        reg = load_registry()
        if not reg:
            return None
        suite = cls(features=reg.get("features", {}))
        for key in BOOSTER_KEYS:
            entry = reg.get("models", {}).get(key)
            if not entry or not model_path(key).exists():
                continue
            try:
                suite.models[key] = load_model(key)
            except Exception as exc:  # a broken artifact must not stop the app
                print(f"[boosting] could not load {key}: {exc}")
                continue
            suite.thresholds[key] = float(entry["threshold"])
            suite.test_metrics[key] = entry.get("test_metrics")
        com = reg.get("models", {}).get(COMMITTEE_KEY)
        if com and len(suite.models) == len(BOOSTER_KEYS):
            suite.thresholds[COMMITTEE_KEY] = float(com["threshold"])
            suite.test_metrics[COMMITTEE_KEY] = com.get("test_metrics")
        return suite if suite.models else None

    @property
    def keys(self) -> list[str]:
        ks = [k for k in BOOSTER_KEYS if k in self.models]
        if COMMITTEE_KEY in self.thresholds:
            ks.append(COMMITTEE_KEY)
        return ks

    @property
    def needs_deep(self) -> bool:
        return bool(self.features.get("use_deep", True))

    def predict_vector(self, x: np.ndarray) -> dict[str, dict]:
        """Score one hybrid feature vector with every booster + the committee."""
        neg = config.CLASS_NAMES[1 - config.POSITIVE_INDEX]
        out: dict[str, dict] = {}
        X = np.asarray(x, dtype="float32").reshape(1, -1)
        for key, model in self.models.items():
            p = float(positive_proba(model, X)[0])
            t = self.thresholds[key]
            out[key] = {"probability": p, "threshold": t,
                        "label": config.POSITIVE_CLASS if p >= t else neg,
                        "display_name": DISPLAY_NAMES[key],
                        "description": DESCRIPTIONS[key]}
        if COMMITTEE_KEY in self.thresholds and all(k in out for k in BOOSTER_KEYS):
            p = float(np.mean([out[k]["probability"] for k in BOOSTER_KEYS]))
            t = self.thresholds[COMMITTEE_KEY]
            out[COMMITTEE_KEY] = {"probability": p, "threshold": t,
                                  "label": config.POSITIVE_CLASS if p >= t else neg,
                                  "display_name": DISPLAY_NAMES[COMMITTEE_KEY],
                                  "description": DESCRIPTIONS[COMMITTEE_KEY]}
        return out
