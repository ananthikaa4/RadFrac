"""Two-model ensemble: HOG+Canny+SVM and MobileNetV2 transfer learning.

Design constraints that shaped this module:

* The two members are **methodologically independent** - hand-crafted gradient
  features fed to an RBF SVM, versus a learned ImageNet representation. That is
  what makes combining them worthwhile at all; two similar CNNs would mostly
  share their mistakes.
* Their raw scores are **not on the same scale**. An SVM's Platt output and a
  sigmoid head's output can both read "0.80" and mean different things, so each
  member is calibrated on the validation split *before* any averaging.
* **Every choice is made on validation data.** Calibration, blend weights and
  the decision threshold are all fitted on val and then frozen into
  ``outputs/models/ensemble.json``. The clean test split is scored exactly once,
  at those frozen settings.

Majority voting is undefined for two members, so "hard voting" is reported as its
two honest variants - AND (both must say fractured) and OR (either suffices) -
rather than inventing a tie-break.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from . import config

MEMBERS = ("classical", "transfer")
MEMBER_LABELS = {
    "classical": "HOG + Canny + SVM",
    "transfer": "MobileNetV2 Transfer Learning",
}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class PlattCalibrator:
    """Platt scaling: a 1-D logistic regression fitted on validation scores.

    Chosen over isotonic regression because it has two parameters instead of a
    step function, so it cannot overfit a 2,000-image validation set, and it is
    monotonic - it never reorders predictions, so ROC-AUC is preserved exactly
    and only the probability *scale* changes.
    """
    coef: float = 1.0
    intercept: float = 0.0
    fitted: bool = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        x = _to_logit(np.asarray(p, dtype="float64")).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x, np.asarray(y).astype(int))
        self.coef = float(lr.coef_[0][0])
        self.intercept = float(lr.intercept_[0])
        self.fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(p, dtype="float64")
        z = self.coef * _to_logit(np.asarray(p, dtype="float64")) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        return cls(coef=d.get("coef", 1.0), intercept=d.get("intercept", 0.0),
                   fitted=d.get("fitted", False))


def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# Frozen configuration
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleConfig:
    members: list = field(default_factory=lambda: list(MEMBERS))
    weights: dict = field(default_factory=dict)          # member -> weight
    calibrators: dict = field(default_factory=dict)      # member -> platt dict
    threshold: float = 0.5
    threshold_source: str = "val-cost-constrained"
    method: str = "weighted-average"
    selected_on: str = "validation"
    val_metrics: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def calibrator(self, member: str) -> PlattCalibrator:
        return PlattCalibrator.from_dict(self.calibrators.get(member, {}))

    def combine(self, probs: dict[str, float | np.ndarray]) -> np.ndarray:
        """Calibrate each member then take the frozen weighted average.

        Missing members are dropped and the remaining weights renormalised, so a
        single-model failure degrades gracefully instead of crashing inference.
        """
        total, acc = 0.0, None
        for m in self.members:
            if m not in probs or probs[m] is None:
                continue
            w = float(self.weights.get(m, 0.0))
            if w <= 0:
                continue
            p = self.calibrator(m).transform(np.atleast_1d(probs[m]))
            acc = p * w if acc is None else acc + p * w
            total += w
        if acc is None or total <= 0:
            return np.array([float("nan")])
        return acc / total

    def save(self, path=None) -> None:
        path = path or config.ENSEMBLE_CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path=None) -> "EnsembleConfig | None":
        path = path or config.ENSEMBLE_CONFIG_FILE
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Metric helper
# --------------------------------------------------------------------------- #
def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    out["roc_auc"] = (float(roc_auc_score(y, p))
                      if len(np.unique(y)) > 1 else None)
    return out


def score_hard(y: np.ndarray, preds: dict[str, np.ndarray], mode: str) -> dict:
    """AND / OR combination of binary member decisions."""
    arrays = [np.asarray(v).astype(int) for v in preds.values()]
    combined = arrays[0].copy()
    for a in arrays[1:]:
        combined = (combined & a) if mode == "and" else (combined | a)
    return score(y, combined.astype(float), 0.5)
