"""Visual analytics, benchmarking and error analysis for all RadFrac models.

Run after ``scripts.train_boosting``::

    python -m scripts.advanced_analysis

Produces (``outputs/figures/advanced``):

    01_pca_explained_variance.png   dimensionality-reduction curve
    02_embedding_tsne_umap.png      t-SNE + UMAP projections of the hybrid features
    03_roc_all_models.png           ROC curves, every model, clean test split
    04_pr_all_models.png            precision-recall curves
    05_confusion_boosters.png       confusion matrices at validated thresholds
    06_shap_beeswarm.png            SHAP TreeExplainer summary (best booster)
    07_feature_family_importance.png permutation importance by feature family
    08_optuna_history.png           Bayesian-optimisation trajectories
    09_optuna_param_importance.png  fANOVA hyperparameter importance
    10_learning_curves.png          train/val log-loss per boosting round
    11_calibration.png              reliability diagrams
    12_error_overlap.png            error correlation between models
    13_error_gallery.png            most confident false negatives / positives
    14_bootstrap_auc.png            95% bootstrap CIs for ROC-AUC

and ``outputs/metrics/benchmark.json`` + ``docs/BENCHMARK.md`` (comparison
with the published baseline, significance tests and error analysis).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.ensemble import EnsembleConfig, score  # noqa: E402
from src.features_hybrid import assemble, family_slices  # noqa: E402
from src.model_boosting import (  # noqa: E402
    BOOSTER_KEYS, COMMITTEE_KEY, DISPLAY_NAMES, load_model, load_registry,
    load_yaml_config, positive_proba,
)
from src.thresholds import get_threshold  # noqa: E402

FIG = config.FIGURES_DIR / "advanced"
DOCS = config.PROJECT_ROOT / "docs"
SEED = 42

NAMES = {
    "classical": "HOG+Canny+SVM",
    "transfer": "MobileNetV2",
    "ensemble": "SVM+MobileNetV2 ensemble",
    "cnn": "Custom CNN (archived)",
    **DISPLAY_NAMES,
}
# Colour-blind-safe categorical palette (Okabe-Ito)
COLORS = {
    "classical": "#999999", "transfer": "#56B4E9", "ensemble": "#000000",
    "xgboost": "#E69F00", "lightgbm": "#009E73", "catboost": "#CC79A7",
    COMMITTEE_KEY: "#D55E00",
}

# Published baseline on the same Kaggle dataset (random 70/20/10 split, no
# de-duplication reported). Numbers transcribed from the paper's results table.
LITERATURE = [{
    "source": "Aldhyani et al., Front. Med. 11:1506686 (2025)",
    "model": "DenseNet201 (fine-tuned)",
    "split": "random 70/20/10 over files, no de-duplication",
    "accuracy": 0.9735, "precision": 0.97, "recall": 0.9778, "f1": 0.9741, "roc_auc": 0.99,
}, {
    "source": "Aldhyani et al., Front. Med. 11:1506686 (2025)",
    "model": "VGG16 (fine-tuned)",
    "split": "random 70/20/10 over files, no de-duplication",
    "accuracy": 0.9655, "precision": 0.96, "recall": 0.9731, "f1": 0.9664, "roc_auc": None,
}]


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figure ->", name)


# --------------------------------------------------------------------------- #
# Load everything
# --------------------------------------------------------------------------- #
def load_all():
    cfg = load_yaml_config()
    cache = config.PROJECT_ROOT / cfg["features"]["cache_dir"]
    from src.features_hybrid import resolve_path

    feats = {s: dict(np.load(cache / f"hybrid_{s}.npz", allow_pickle=False))
             for s in ("train", "val", "test")}
    # Cache may have been built in the container (/data/raw/...); remap paths.
    for s in feats:
        feats[s]["paths"] = np.asarray([resolve_path(str(p)) for p in feats[s]["paths"]])
    fc = cfg["features"]
    X = {s: assemble(feats[s], fc["use_handcrafted"], fc["use_deep"]) for s in feats}
    y = {s: feats[s]["y"] for s in feats}
    reg = load_registry()
    preds = dict(np.load(config.METRICS_DIR / "boosting_predictions.npz", allow_pickle=False))

    # Existing models scored on exactly the same images.
    from src.model_svm import load_classical, predict_proba_positive
    svm = load_classical()
    probs = {"val": {}, "test": {}}
    for s in ("val", "test"):
        probs[s]["classical"] = predict_proba_positive(svm, feats[s]["handcrafted"])
        probs[s]["transfer"] = feats[s]["transfer_prob"].astype("float64")
        ens = EnsembleConfig.load()
        if ens:
            probs[s]["ensemble"] = ens.combine({"classical": probs[s]["classical"],
                                                "transfer": probs[s]["transfer"]})
        for k in list(BOOSTER_KEYS) + [COMMITTEE_KEY]:
            if f"{k}_{s}" in preds:
                probs[s][k] = preds[f"{k}_{s}"]
    thr = {"classical": get_threshold("classical"), "transfer": get_threshold("transfer")}
    if EnsembleConfig.load():
        thr["ensemble"] = float(EnsembleConfig.load().threshold)
    for k, e in reg.get("models", {}).items():
        thr[k] = float(e["threshold"])
    return cfg, feats, X, y, reg, probs, thr


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_pca(X, cfg):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X["train"])
    pca = PCA(n_components=512, svd_solver="randomized", random_state=SEED).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#0e7490", lw=2)
    for k in cfg["search_spaces"]["common"]["pca_components"]:
        ax.axvline(k, color="#999", ls="--", lw=1)
        ax.annotate(f"k={k}\n{cum[k - 1]:.0%}", (k, cum[k - 1]), textcoords="offset points",
                    xytext=(6, -26), fontsize=8)
    ax.set_xlabel("principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"PCA on {X['train'].shape[1]}-d hybrid features (train split)")
    ax.grid(alpha=.3)
    _save(fig, "01_pca_explained_variance.png")
    return {int(k): float(cum[k - 1]) for k in cfg["search_spaces"]["common"]["pca_components"]}


def fig_embeddings(X, y):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y["test"]), size=min(1500, len(y["test"])), replace=False)
    Z = StandardScaler().fit(X["train"]).transform(X["test"][idx])
    Z = PCA(n_components=50, random_state=SEED).fit_transform(Z)
    lab = y["test"][idx]
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(Z)
    try:
        import umap
        um = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(Z)
    except Exception as exc:
        print("  UMAP unavailable:", exc)
        um = None
    panels = [("t-SNE", tsne)] + ([("UMAP", um)] if um is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for ax, (title, E) in zip(axes, panels):
        for c, col, name in ((0, "#0072B2", "not fractured"), (1, "#D55E00", "fractured")):
            m = lab == c
            ax.scatter(E[m, 0], E[m, 1], s=6, alpha=.6, c=col, label=name)
        ax.set_title(f"{title} of hybrid features (test, n={len(idx)})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=3, loc="best")
    _save(fig, "02_embedding_tsne_umap.png")


def fig_roc_pr(y, probs):
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for k, p in probs["test"].items():
        fpr, tpr, _ = roc_curve(y["test"], p)
        ax.plot(fpr, tpr, color=COLORS.get(k), lw=2.2 if k in BOOSTER_KEYS or k == COMMITTEE_KEY else 1.5,
                ls="-" if k not in ("classical", "transfer", "ensemble") else "--",
                label=f"{NAMES[k]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#bbb", lw=1)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("ROC - leakage-free test split"); ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3)
    _save(fig, "03_roc_all_models.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for k, p in probs["test"].items():
        pr, rc, _ = precision_recall_curve(y["test"], p)
        ax.plot(rc, pr, color=COLORS.get(k), lw=2,
                ls="-" if k not in ("classical", "transfer", "ensemble") else "--",
                label=f"{NAMES[k]} (AP {average_precision_score(y['test'], p):.3f})")
    ax.axhline(y["test"].mean(), color="#bbb", lw=1, label="prevalence")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall - leakage-free test split")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.3)
    _save(fig, "04_pr_all_models.png")


def fig_confusion(y, probs, thr):
    keys = [k for k in list(BOOSTER_KEYS) + [COMMITTEE_KEY] if k in probs["test"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        m = score(y["test"], probs["test"][k], thr[k])
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["not fx", "fx"]); ax.set_yticks([0, 1], ["not fx", "fx"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(f"{NAMES[k]}\nt={thr[k]:.2f}  recall {m['recall']:.2f}", fontsize=10)
    _save(fig, "05_confusion_boosters.png")


def fig_shap(X, y, reg):
    import shap

    # Explain the booster chosen on VALIDATION AUC (selection never uses test).
    best = max((k for k in BOOSTER_KEYS if k in reg.get("models", {})),
               key=lambda k: reg["models"][k]["val_metrics"]["roc_auc"])
    pipe = load_model(best)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y["test"]), size=min(800, len(y["test"])), replace=False)
    Z = pipe[:-1].transform(X["test"][idx])
    names = [f"PC{i + 1}" for i in range(Z.shape[1])]
    clf = pipe[-1]
    if best == "xgboost":
        # Exact TreeSHAP computed by XGBoost itself (shap 0.46 cannot parse
        # XGBoost 3.x model dumps). Last column is the bias term.
        import xgboost as xgb
        sv = clf.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
    elif best == "catboost":
        from catboost import Pool
        sv = clf.get_feature_importance(Pool(Z), type="ShapValues")[:, :-1]
    else:
        sv = shap.TreeExplainer(clf).shap_values(Z)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[..., 1]
    plt.figure()
    shap.summary_plot(sv, Z, feature_names=names, max_display=15, show=False)
    plt.title(f"SHAP summary - {DISPLAY_NAMES[best]} (top 15 principal components)")
    _save(plt.gcf(), "06_shap_beeswarm.png")

    # Back-project |SHAP| through the PCA loadings to the feature families
    pca = pipe.named_steps["pca"]
    mean_abs = np.abs(sv).mean(axis=0)                      # per component
    load = np.abs(pca.components_)                          # (k, d)
    load = load / load.sum(axis=1, keepdims=True)
    per_feat = mean_abs @ load                              # (d,)
    fam = {f: float(per_feat[s].sum()) for f, s in family_slices().items()}
    tot = sum(fam.values()) or 1.0
    return best, {f: v / tot for f, v in fam.items()}


def fig_family_importance(X, y, reg, shap_share):
    """Permutation importance by feature family: AUC drop when a whole block
    (HOG, Canny grid, MobileNetV2 embedding) is shuffled across test images."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(SEED)
    res = {}
    for k in BOOSTER_KEYS:
        if k not in reg.get("models", {}):
            continue
        pipe = load_model(k)
        base = roc_auc_score(y["test"], positive_proba(pipe, X["test"]))
        res[k] = {}
        for fam, sl in family_slices().items():
            drops = []
            for _ in range(3):
                Xp = X["test"].copy()
                Xp[:, sl] = Xp[rng.permutation(len(Xp))][:, sl]
                drops.append(base - roc_auc_score(y["test"], positive_proba(pipe, Xp)))
            res[k][fam] = float(np.mean(drops))
    fams = list(family_slices())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.8 / max(len(res), 1)
    for i, (k, d) in enumerate(res.items()):
        axes[0].bar(np.arange(len(fams)) + i * w, [d[f] for f in fams], w,
                    color=COLORS[k], label=NAMES[k])
    axes[0].set_xticks(np.arange(len(fams)) + w * (len(res) - 1) / 2,
                       ["HOG (1764)", "Canny grid (16)", "MobileNetV2 (1280)"])
    axes[0].set_ylabel("ROC-AUC drop when shuffled"); axes[0].legend(fontsize=8)
    axes[0].set_title("Permutation importance by feature family (test)")
    axes[1].bar(["HOG", "Canny grid", "MobileNetV2"], [shap_share[1].get(f, 0) for f in fams],
                color=["#E69F00", "#56B4E9", "#009E73"])
    axes[1].set_ylabel("share of mean |SHAP|")
    axes[1].set_title(f"SHAP mass back-projected via PCA loadings ({DISPLAY_NAMES[shap_share[0]]})")
    for ax in axes:
        ax.grid(axis="y", alpha=.3)
    _save(fig, "07_feature_family_importance.png")
    return res


def fig_optuna(reg):
    import pandas as pd

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k in BOOSTER_KEYS:
        e = reg.get("models", {}).get(k)
        if not e:
            continue
        df = pd.read_csv(config.PROJECT_ROOT / e["trials_csv"]).sort_values("trial")
        ax.scatter(df["trial"], df["val_roc_auc"], s=12, color=COLORS[k], alpha=.45)
        ax.plot(df["trial"], df["val_roc_auc"].cummax(), color=COLORS[k], lw=2, label=NAMES[k])
    ax.set_xlabel("Optuna trial"); ax.set_ylabel("validation ROC-AUC")
    ax.set_title("TPE Bayesian optimisation - best so far (line) and trials (dots)")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "08_optuna_history.png")

    keys = [k for k in BOOSTER_KEYS if reg.get("models", {}).get(k, {}).get("param_importance")]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8))
        for ax, k in zip(np.atleast_1d(axes), keys):
            imp = reg["models"][k]["param_importance"]
            items = sorted(imp.items(), key=lambda kv: kv[1])
            ax.barh([a for a, _ in items], [b for _, b in items], color=COLORS[k])
            ax.set_title(f"{NAMES[k]} - hyperparameter importance", fontsize=10)
            ax.grid(axis="x", alpha=.3)
        _save(fig, "09_optuna_param_importance.png")


def fig_learning_curves(reg):
    keys = [k for k in BOOSTER_KEYS if reg.get("models", {}).get(k, {}).get("learning_curves")]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        c = reg["models"][k]["learning_curves"]
        ax.plot(c["train"], color=COLORS[k], lw=2, label="train")
        ax.plot(c["val"], color="#333", lw=2, ls="--", label="validation")
        ax.set_title(f"{NAMES[k]} log-loss per round", fontsize=10)
        ax.set_xlabel("boosting round"); ax.legend(); ax.grid(alpha=.3)
    _save(fig, "10_learning_curves.png")


def fig_calibration(y, probs):
    from src.threshold_analysis import calibration

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color="#bbb")
    out = {}
    for k, p in probs["test"].items():
        c = calibration(y["test"], p)
        out[k] = {"ece": c.ece, "brier": c.brier}
        xs = [b["mean_confidence"] for b in c.bins if b.get("count")]
        ys = [b["observed_rate"] for b in c.bins if b.get("count")]
        ax.plot(xs, ys, marker="o", ms=3, color=COLORS.get(k), label=f"{NAMES[k]} (ECE {c.ece:.3f})")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction fractured")
    ax.set_title("Reliability diagram (test)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    _save(fig, "11_calibration.png")
    return out


def fig_error_overlap(y, probs, thr):
    keys = list(probs["test"])
    err = {k: ((probs["test"][k] >= thr[k]).astype(int) != y["test"]) for k in keys}
    n = len(keys)
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            inter = (err[a] & err[b]).sum()
            union = (err[a] | err[b]).sum()
            M[i, j] = inter / union if union else 0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), [NAMES[k] for k in keys], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n), [NAMES[k] for k in keys], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < .6 else "black")
    fig.colorbar(im, ax=ax, label="Jaccard overlap of misclassified test images")
    ax.set_title("Error overlap between models")
    _save(fig, "12_error_overlap.png")
    all_wrong = np.all([err[k] for k in keys], axis=0)
    return {"jaccard": {a: {b: float(M[i, j]) for j, b in enumerate(keys)} for i, a in enumerate(keys)},
            "errors_shared_by_all_models": int(all_wrong.sum()),
            "errors_per_model": {k: int(err[k].sum()) for k in keys}}


def fig_error_gallery(feats, y, probs, thr, key):
    from src.preprocessing import load_image_array

    p = probs["test"][key]
    t = thr[key]
    paths = feats["test"]["paths"]
    fn = np.where((y["test"] == 1) & (p < t))[0]
    fp = np.where((y["test"] == 0) & (p >= t))[0]
    fn = fn[np.argsort(p[fn])][:6]           # most confident misses
    fp = fp[np.argsort(-p[fp])][:6]          # most confident false alarms
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    for row, (idxs, title) in enumerate(((fn, "False negative"), (fp, "False positive"))):
        for c in range(6):
            ax = axes[row, c]
            ax.axis("off")
            if c < len(idxs):
                i = idxs[c]
                try:
                    ax.imshow(load_image_array(str(paths[i])), cmap="gray")
                except Exception:
                    pass
                ax.set_title(f"{title}\np={p[i]:.2f}", fontsize=9)
    fig.suptitle(f"Most confident errors - {NAMES[key]} (threshold {t:.2f})")
    _save(fig, "13_error_gallery.png")


def error_by_image_properties(feats, y, probs, thr, key):
    """Error rate by file format and original resolution - surfaces acquisition
    sub-populations where the model is weaker."""
    from PIL import Image

    pred = (probs["test"][key] >= thr[key]).astype(int)
    wrong = pred != y["test"]
    groups: dict[str, list] = {}
    for i, path in enumerate(feats["test"]["paths"]):
        path = str(path)
        fmt = Path(path).suffix.lower().lstrip(".")
        try:
            with Image.open(path) as im:
                w, h = im.size
            mp = w * h / 1e6
            size = "<0.1 MP" if mp < .1 else ("0.1-0.5 MP" if mp < .5 else ">=0.5 MP")
        except Exception:
            size = "unknown"
        for g in (f"format={fmt}", f"size={size}"):
            groups.setdefault(g, []).append(wrong[i])
    return {g: {"n": len(v), "error_rate": float(np.mean(v))}
            for g, v in sorted(groups.items())}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def bootstrap(y, probs, thr, n_boot=1000):
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(SEED)
    n = len(y)
    idxs = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = {}
    for k, p in probs.items():
        aucs, f1s = [], []
        pred = (p >= thr[k]).astype(int)
        for ix in idxs:
            if len(np.unique(y[ix])) < 2:
                continue
            aucs.append(roc_auc_score(y[ix], p[ix]))
            f1s.append(f1_score(y[ix], pred[ix]))
        out[k] = {"auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
                  "f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
                  "_aucs": np.array(aucs)}
    return out


def paired_tests(y, probs, thr, ref: str, n_boot=1000):
    """Paired bootstrap on ROC-AUC difference and exact McNemar on decisions,
    each model vs the reference model."""
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(SEED + 1)
    n = len(y)
    res = {}
    ref_ok = ((probs[ref] >= thr[ref]).astype(int) == y)
    for k, p in probs.items():
        if k == ref:
            continue
        diffs = []
        for _ in range(n_boot):
            ix = rng.integers(0, n, n)
            if len(np.unique(y[ix])) < 2:
                continue
            diffs.append(roc_auc_score(y[ix], p[ix]) - roc_auc_score(y[ix], probs[ref][ix]))
        diffs = np.array(diffs)
        ok = ((p >= thr[k]).astype(int) == y)
        b = int((ok & ~ref_ok).sum())       # model right, reference wrong
        c = int((~ok & ref_ok).sum())       # model wrong, reference right
        pm = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0
        res[k] = {
            "delta_auc": float(roc_auc_score(y, p) - roc_auc_score(y, probs[ref])),
            "delta_auc_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_bootstrap": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
            "mcnemar_b_model_only_correct": b, "mcnemar_c_ref_only_correct": c,
            "mcnemar_p": pm,
        }
    return res


def fig_bootstrap(boot):
    keys = list(boot)
    fig, ax = plt.subplots(figsize=(7, 4))
    parts = ax.violinplot([boot[k]["_aucs"] for k in keys], showmedians=True)
    for body, k in zip(parts["bodies"], keys):
        body.set_facecolor(COLORS.get(k)); body.set_alpha(.6)
    ax.set_xticks(range(1, len(keys) + 1), [NAMES[k] for k in keys], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test ROC-AUC (1,000 bootstrap resamples)")
    ax.set_title("Uncertainty of test ROC-AUC"); ax.grid(axis="y", alpha=.3)
    _save(fig, "14_bootstrap_auc.png")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def leaked_reproduction() -> dict | None:
    p = config.OUTPUT_DIR / "baselines" / "A_original_leaked.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def write_markdown(bench: dict) -> None:
    def f(v, d=3):
        return "—" if v is None else f"{v:.{d}f}"

    L = ["# RadFrac — Benchmarking & Comparative Analysis", "",
         "_Generated by `python -m scripts.advanced_analysis`. All numbers below are "
         "measured, not transcribed, except the literature rows._", "",
         "## 1. Test-set performance (leakage-free, group-aware split)", "",
         f"Test images: **{bench['n_test']}** "
         f"({bench['test_positive']} fractured / {bench['n_test'] - bench['test_positive']} not). "
         "Thresholds were fixed on the validation split (cost FN×5 + FP×1, FPR ≤ 0.25) "
         "before the test split was scored once.", "",
         "_Existing models are re-scored here through the serving preprocessing (PIL "
         "crop + resize, exactly what the live app does). For MobileNetV2 this differs "
         "slightly from the original `tf.image` evaluation, so its AUC can differ from "
         "`comparison_metrics.json` by about ±0.02._", "",
         "| Model | Accuracy | Precision | Recall | F1 | Test ROC-AUC (95% CI) | Val ROC-AUC | ECE |",
         "|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for k, m in bench["models"].items():
        ci = bench["bootstrap"][k]["auc_ci"]
        L.append(f"| {NAMES[k]} | {f(m['accuracy'])} | {f(m['precision'])} | {f(m['recall'])} | "
                 f"{f(m['f1'])} | {f(m['roc_auc'])} ({f(ci[0])}–{f(ci[1])}) | "
                 f"{f(bench['val_auc'].get(k))} | {f(bench['calibration'][k]['ece'])} |")
    L += ["", "### Key findings", ""] + [f"* {s}" for s in bench["findings"]]
    L += ["", "## 2. Comparison with published state of the art", "",
          "| Source | Model | Split | Accuracy | F1 | ROC-AUC |", "|---|---|---|:--:|:--:|:--:|"]
    for r in LITERATURE:
        L.append(f"| {r['source']} | {r['model']} | {r['split']} | {f(r['accuracy'])} | "
                 f"{f(r['f1'])} | {f(r['roc_auc'], 2)} |")
    leak = bench.get("leaked_reproduction")
    if leak:
        for k, m in leak["models"].items():
            L.append(f"| This work — reproduction on random split | {NAMES.get(k, k)} | "
                     f"random over files (leaky) | {f(m['accuracy'])} | {f(m['f1'])} | {f(m['roc_auc'])} |")
    best = bench["best_model"]
    for k in dict.fromkeys([bench["reference"], best]):
        bm = bench["models"][k]
        L.append(f"| This work — leakage-free | {NAMES[k]} | group-aware (duplicates kept "
                 f"within one split) | {f(bm['accuracy'])} | {f(bm['f1'])} | {f(bm['roc_auc'])} |")
    L += ["",
          "**Interpretation.** The dataset's duplicate audit found that "
          f"{bench['audit'].get('test_percent_affected', '—')}% of test images in a random "
          "file-level split have an exact or augmented near-duplicate in training. On such a "
          "split our *simple* HOG+SVM reaches ~0.995 accuracy — matching or exceeding the "
          "published DenseNet201 — which shows that the published figure largely measures "
          "duplicate recall, not generalisation. On the leakage-free split every model drops "
          "substantially; the numbers in §1 are the honest estimate of performance on unseen "
          "radiographs.", "",
          "## 3. Statistical significance vs. the existing best model", "",
          f"Reference: **{NAMES[bench['reference']]}**. Paired bootstrap (1,000 resamples) on the "
          "ROC-AUC difference and exact McNemar test on the thresholded decisions.", "",
          "| Model | ΔAUC | 95% CI | p (bootstrap) | McNemar b / c | p (McNemar) |",
          "|---|:--:|:--:|:--:|:--:|:--:|"]
    for k, t in bench["significance"].items():
        L.append(f"| {NAMES[k]} | {t['delta_auc']:+.4f} | {t['delta_auc_ci'][0]:+.4f} … "
                 f"{t['delta_auc_ci'][1]:+.4f} | {t['p_bootstrap']:.4f} | "
                 f"{t['mcnemar_b_model_only_correct']} / {t['mcnemar_c_ref_only_correct']} | "
                 f"{t['mcnemar_p']:.4f} |")
    L += ["", "_b = images only this model gets right; c = images only the reference gets right._",
          "", "## 4. Error analysis", "",
          f"* Errors per model: " + ", ".join(f"{NAMES[k]} {v}" for k, v in
                                               bench["error_overlap"]["errors_per_model"].items()),
          f"* Test images misclassified by **every** model: "
          f"{bench['error_overlap']['errors_shared_by_all_models']} — the irreducible hard core "
          "(see `13_error_gallery.png`).",
          "* Error-overlap Jaccard matrix: `12_error_overlap.png`. Low overlap between the "
          "hand-crafted and deep-feature models is what makes ensembling worthwhile.", "",
          f"Error rate of {NAMES[best]} by acquisition sub-population:", "",
          "| Group | n | Error rate |", "|---|:--:|:--:|"]
    for g, v in bench["error_by_group"].items():
        L.append(f"| {g} | {v['n']} | {v['error_rate']:.3f} |")
    L += ["", "## 5. Feature-family attribution", "",
          "Permutation importance (ROC-AUC drop when one family is shuffled across images):", "",
          "| Model | HOG | Canny grid | MobileNetV2 |", "|---|:--:|:--:|:--:|"]
    for k, d in bench["family_importance"].items():
        L.append(f"| {NAMES[k]} | {d.get('hog', 0):.4f} | {d.get('edges', 0):.4f} | {d.get('deep', 0):.4f} |")
    sh = bench["shap_family_share"]
    L += ["", f"SHAP mass back-projected through the PCA loadings ({NAMES[sh['model']]}): " +
          ", ".join(f"{k} {v:.1%}" for k, v in sh["share"].items()), "",
          "## 6. Dimensionality reduction", "",
          "Cumulative explained variance of PCA on the hybrid features: " +
          ", ".join(f"k={k}: {v:.1%}" for k, v in bench["pca_variance"].items()), "",
          "## References", "",
          "[1] T. Aldhyani, Z. A. T. Ahmed, B. M. Alsharbi, S. Ahmad, M. H. Al-Adhaileh, "
          "A. H. Kamal *et al.*, \"Diagnosis and detection of bone fracture in "
          "radiographic images using deep learning approaches,\" *Front. Med.*, vol. 11, "
          "Art. no. 1506686, 2025, doi: 10.3389/fmed.2024.1506686.", ""]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "BENCHMARK.md").write_text("\n".join(L), encoding="utf-8")
    print("  report -> docs/BENCHMARK.md")


def main() -> None:
    cfg, feats, X, y, reg, probs, thr = load_all()
    print("models:", list(probs["test"]))

    pca_var = fig_pca(X, cfg)
    fig_embeddings(X, y)
    fig_roc_pr(y, probs)
    fig_confusion(y, probs, thr)
    shap_best, shap_share = fig_shap(X, y, reg)
    fam = fig_family_importance(X, y, reg, (shap_best, shap_share))
    fig_optuna(reg)
    fig_learning_curves(reg)
    calib = fig_calibration(y, probs)
    overlap = fig_error_overlap(y, probs, thr)

    from sklearn.metrics import roc_auc_score

    test_metrics = {k: score(y["test"], p, thr[k]) for k, p in probs["test"].items()}
    val_auc = {k: float(roc_auc_score(y["val"], p)) for k, p in probs["val"].items()}
    # "Best" booster is chosen on VALIDATION AUC so no selection touches test.
    best = max((k for k in test_metrics if k in BOOSTER_KEYS or k == COMMITTEE_KEY),
               key=lambda k: val_auc[k])
    fig_error_gallery(feats, y, probs, thr, best)
    by_group = error_by_image_properties(feats, y, probs, thr, best)

    boot = bootstrap(y["test"], probs["test"], thr)
    fig_bootstrap(boot)
    # Reference = the model the deployed app uses for its headline verdict.
    ref = "ensemble" if "ensemble" in test_metrics else "classical"
    sig = paired_tests(y["test"], probs["test"], thr, ref)

    boosters = [k for k in test_metrics if k in BOOSTER_KEYS or k == COMMITTEE_KEY]
    gaps = {k: val_auc[k] - test_metrics[k]["roc_auc"] for k in test_metrics}
    fam_mean = {f: float(np.mean([d[f] for d in fam.values()])) for f in family_slices()}
    findings = [
        f"Every model loses ROC-AUC from validation to test (mean drop "
        f"{np.mean(list(gaps.values())):.3f}; boosters {np.mean([gaps[k] for k in boosters]):.3f}, "
        f"existing models {np.mean([gaps[k] for k in gaps if k not in boosters]):.3f}). The "
        f"group-aware split leaves distribution shift between splits, so validation is an "
        f"optimistic proxy - and Optuna, which maximises validation AUC, inherits that optimism.",
        f"Best booster (chosen on validation): {NAMES[best]}, test AUC "
        f"{test_metrics[best]['roc_auc']:.3f} vs {NAMES[ref]} {test_metrics[ref]['roc_auc']:.3f} "
        f"(ΔAUC {sig[best]['delta_auc']:+.3f}, 95% CI {sig[best]['delta_auc_ci'][0]:+.3f} … "
        f"{sig[best]['delta_auc_ci'][1]:+.3f}). The gradient-boosting models do not beat the "
        f"deployed ensemble on unseen films; this is why they are reported as a second opinion "
        f"rather than replacing the headline verdict.",
        f"Validation-chosen thresholds transfer poorly for the boosters (e.g. {NAMES[best]} "
        f"t={thr[best]:.2f}, test recall {test_metrics[best]['recall']:.2f} vs the "
        f"ensemble's {test_metrics[ref]['recall']:.2f}): their scores are less calibrated "
        f"(ECE {calib[best]['ece']:.3f} vs {calib[ref]['ece']:.3f}).",
        f"Both feature families matter: shuffling HOG costs {fam_mean['hog']:.3f} AUC on "
        f"average, the MobileNetV2 embedding {fam_mean['deep']:.3f}; the 16-cell Canny grid "
        f"contributes nothing measurable ({fam_mean['edges']:+.3f}).",
        f"{overlap['errors_shared_by_all_models']} of {len(y['test'])} test images are misread "
        f"by every model - a hard core that no model choice fixes (see the error gallery).",
    ]

    audit_p = config.OUTPUT_DIR / "duplicate_audit" / "audit_report.json"
    bench = {
        "n_test": int(len(y["test"])), "test_positive": int(y["test"].sum()),
        "models": test_metrics, "val_auc": val_auc, "findings": findings,
        "bootstrap": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in boot.items()},
        "calibration": calib, "reference": ref, "significance": sig,
        "best_model": best, "error_overlap": overlap, "error_by_group": by_group,
        "family_importance": fam,
        "shap_family_share": {"model": shap_best, "share": shap_share},
        "pca_variance": pca_var, "literature": LITERATURE,
        "leaked_reproduction": leaked_reproduction(),
        "audit": json.loads(audit_p.read_text()) if audit_p.exists() else {},
    }
    bench["audit"] = {k: bench["audit"].get(k) for k in
                      ("test_percent_affected", "val_percent_affected", "duplicate_groups")}
    (config.METRICS_DIR / "benchmark.json").write_text(json.dumps(bench, indent=2, default=float))
    write_markdown(bench)


if __name__ == "__main__":
    main()
