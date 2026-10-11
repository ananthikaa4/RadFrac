"""Cost-sensitive threshold selection and calibration analysis (section 3, 4.4).

Two distinct questions, both essential to the clinical-trust narrative:

**1. Where should the decision threshold sit?**
At a screening step, a missed fracture (false negative) is far costlier than a
false alarm (false positive): the false alarm costs one radiologist review, the
miss can cost a mis-managed injury. Maximising accuracy or even F1 implicitly
assumes symmetric costs, which is simply wrong here. We therefore sweep the
threshold and minimise a weighted cost

    cost(t) = FN(t) * FALSE_NEGATIVE_COST + FP(t) * FALSE_POSITIVE_COST

with the weights configurable (defaults 5:1). We also report the smallest
threshold achieving a target recall, which is how a screening tool is usually
specified in practice.

**2. Is the reported confidence honest?**
A doctor needs "90% confident" to mean "correct about 90% of the time". We
compute a reliability curve plus Expected Calibration Error (ECE) and Brier
score. Poor calibration does not make a model useless for ranking, but it must
be disclosed rather than presented as a probability.

All threshold selection happens on the **validation** split; the test split is
only ever used for the final reporting pass, so no operating point is tuned on
test data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Cost-sensitive threshold sweep
# --------------------------------------------------------------------------- #
@dataclass
class CostAnalysis:
    best_threshold: float
    best_cost: float
    fn_cost: float
    fp_cost: float
    at_best: dict                 # tp/fp/fn/tn, precision, recall, f1 at best t
    recall_target: float | None = None
    threshold_for_recall: float | None = None
    curve: list[dict] | None = None
    max_fpr: float | None = None
    constraint_active: bool = False
    unconstrained_threshold: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def sweep_costs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float | None = None,
    fp_cost: float | None = None,
    recall_target: float | None = 0.95,
    n_points: int = 199,
    max_fpr: float | None = None,
) -> CostAnalysis:
    """Sweep thresholds and pick the weighted-cost minimiser.

    The search is restricted to thresholds whose false-positive rate is at most
    ``max_fpr``. Without that constraint the minimiser degenerates to "predict
    positive for everything" whenever the model is weak and false negatives are
    weighted heavily - see ``config.MAX_FALSE_POSITIVE_RATE``.
    """
    fn_cost = config.FALSE_NEGATIVE_COST if fn_cost is None else fn_cost
    fp_cost = config.FALSE_POSITIVE_COST if fp_cost is None else fp_cost
    max_fpr = config.MAX_FALSE_POSITIVE_RATE if max_fpr is None else max_fpr

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")

    grid = np.linspace(0.005, 0.995, n_points)
    curve: list[dict] = []
    best_t, best_cost, best_stats = config.DEFAULT_THRESHOLD, float("inf"), {}
    unc_t, unc_cost = None, float("inf")

    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        tp, fp, fn, tn = _counts(y_true, y_pred)
        cost = fn * fn_cost + fp * fp_cost
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        row = {
            "threshold": round(float(t), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "fpr": round(fpr, 4), "specificity": round(1.0 - fpr, 4),
            "f1": round(f1, 4), "cost": round(float(cost), 4),
        }
        curve.append(row)

        # Track the unconstrained optimum purely for reporting.
        if cost < unc_cost:
            unc_cost, unc_t = float(cost), float(t)

        if fpr > max_fpr:                       # violates the specificity floor
            continue
        # Tie-break toward higher recall (safer at a screening step).
        if cost < best_cost or (cost == best_cost and recall > best_stats.get("recall", -1)):
            best_cost, best_t, best_stats = float(cost), float(t), row

    constraint_active = (unc_t is not None and abs(unc_t - best_t) > 1e-9)
    note = ""
    if not best_stats:
        # Nothing met the constraint: fall back to the highest-specificity point.
        feasible = min(curve, key=lambda r: r["fpr"])
        best_stats = feasible
        best_t, best_cost = feasible["threshold"], feasible["cost"]
        note = (f"no threshold achieved FPR <= {max_fpr:.2f}; fell back to the "
                f"highest-specificity point (FPR={feasible['fpr']:.3f})")
    elif constraint_active:
        note = (f"specificity constraint binding: unconstrained cost optimum was "
                f"t={unc_t:.3f}, which exceeded the FPR ceiling of {max_fpr:.2f}")

    # Smallest threshold (i.e. most permissive) still hitting the recall target.
    thr_for_recall = None
    if recall_target is not None:
        hits = [r for r in curve if r["recall"] >= recall_target]
        if hits:
            thr_for_recall = max(r["threshold"] for r in hits)

    return CostAnalysis(
        best_threshold=best_t, best_cost=best_cost,
        fn_cost=fn_cost, fp_cost=fp_cost, at_best=best_stats,
        recall_target=recall_target, threshold_for_recall=thr_for_recall,
        curve=curve, max_fpr=max_fpr, constraint_active=constraint_active,
        unconstrained_threshold=unc_t, note=note,
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    ece: float                     # Expected Calibration Error
    mce: float                     # Maximum Calibration Error
    brier: float
    bins: list[dict]
    n: int
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    """Reliability-curve statistics: does confidence match correctness?"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    n = int(y_true.size)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lower": round(float(lo), 3), "upper": round(float(hi), 3),
                         "count": 0, "mean_confidence": None, "observed_rate": None,
                         "gap": None})
            continue
        conf = float(y_prob[mask].mean())
        obs = float(y_true[mask].mean())
        gap = abs(conf - obs)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append({
            "lower": round(float(lo), 3), "upper": round(float(hi), 3),
            "count": count, "mean_confidence": round(conf, 4),
            "observed_rate": round(obs, 4), "gap": round(gap, 4),
        })

    brier = float(np.mean((y_prob - y_true) ** 2))
    if ece < 0.05:
        verdict = "well calibrated - reported confidence can be read as a probability"
    elif ece < 0.15:
        verdict = "moderately miscalibrated - treat confidence as a ranking score"
    else:
        verdict = ("poorly calibrated - the raw score is NOT a probability and must "
                   "not be presented to a clinician as one")

    return CalibrationResult(ece=round(ece, 4), mce=round(mce, 4),
                             brier=round(brier, 4), bins=bins, n=n,
                             verdict=verdict)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_cost_curve(analyses: dict[str, CostAnalysis], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, a in analyses.items():
        ts = [r["threshold"] for r in a.curve]
        axes[0].plot(ts, [r["cost"] for r in a.curve], label=f"{name}")
        axes[0].axvline(a.best_threshold, ls=":", alpha=0.5)
        axes[1].plot([r["recall"] for r in a.curve],
                     [r["precision"] for r in a.curve], label=name)
    axes[0].set_xlabel("decision threshold")
    axes[0].set_ylabel(f"weighted cost (FN×{config.FALSE_NEGATIVE_COST:g}, "
                       f"FP×{config.FALSE_POSITIVE_COST:g})")
    axes[0].set_title("Cost-sensitive threshold sweep")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("recall (sensitivity)")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall trade-off")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(results: dict[str, CalibrationResult], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
    for name, r in results.items():
        xs = [b["mean_confidence"] for b in r.bins if b["count"]]
        ys = [b["observed_rate"] for b in r.bins if b["count"]]
        ax.plot(xs, ys, marker="o", ms=4, label=f"{name} (ECE={r.ece:.3f})")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed fracture rate")
    ax.set_title("Calibration (reliability) curve")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_cost_thresholds(mapping: dict[str, float]) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.COST_THRESHOLDS_FILE.write_text(json.dumps(mapping, indent=2))


def load_cost_thresholds() -> dict[str, float]:
    if config.COST_THRESHOLDS_FILE.exists():
        return json.loads(config.COST_THRESHOLDS_FILE.read_text())
    return {}
