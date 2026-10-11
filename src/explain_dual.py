"""Dual explainability with cross-method agreement (4.5).

A single saliency method is easy to over-trust: Grad-CAM can look convincing
even when a model is keying on an artefact. Two *methodologically independent*
attributions that agree on the same region is a much stronger claim than either
alone - and when they disagree, that is itself a useful signal to flag.

Methods used
------------
* **Grad-CAM** - gradient-based, uses the model's internal conv activations.
* **Occlusion sensitivity** - purely black-box: slide a patch over the image,
  measure how much the fracture score drops. Model-agnostic, so it also works
  for the classical SVM where Grad-CAM does not apply.

Occlusion is preferred over KernelSHAP/LIME here for a deliberate reason: on a
128x128 radiograph both SHAP and LIME operate on superpixel segmentations whose
randomness makes the agreement metric unstable run-to-run, and both need
hundreds of model evaluations. Occlusion is deterministic, needs no segmentation
hyperparameters, and is directly interpretable as "score drop when this region
is hidden" - a property a clinician can reason about. (``shap`` is installed and
``kernel_shap_attribution`` is provided for users who want it.)

Agreement metrics
-----------------
* **Pearson correlation** between the two normalised maps.
* **IoU** of their top-attention regions (above a percentile cut).
* **Peak distance** between the two attention centroids, normalised by image
  diagonal.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config


@dataclass
class AgreementResult:
    correlation: float
    iou: float
    peak_distance: float          # 0 = same peak, 1 = opposite corners
    agree: bool
    verdict: str
    conclusive: bool = True       # False when a map carries no usable signal

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Occlusion sensitivity (model-agnostic)
# --------------------------------------------------------------------------- #
def occlusion_attribution(
    predict_fn,
    img_uint8: np.ndarray,
    patch: int = 24,
    stride: int = 12,
    fill: str = "mean",
) -> np.ndarray:
    """Attribution map from sliding-window occlusion.

    ``predict_fn`` takes an RGB uint8 image and returns P(fractured). Higher map
    values mean hiding that region reduced the score more, i.e. it mattered more.
    """
    img = img_uint8
    h, w = img.shape[:2]
    base = float(predict_fn(img))

    fill_value = (float(img.mean()) if fill == "mean" else 0.0)
    heat = np.zeros((h, w), dtype="float64")
    counts = np.zeros((h, w), dtype="float64")

    for y in range(0, max(h - patch + 1, 1), stride):
        for x in range(0, max(w - patch + 1, 1), stride):
            occluded = img.copy()
            occluded[y:y + patch, x:x + patch] = fill_value
            drop = base - float(predict_fn(occluded))
            heat[y:y + patch, x:x + patch] += drop
            counts[y:y + patch, x:x + patch] += 1.0

    counts[counts == 0] = 1.0
    heat /= counts
    heat = np.clip(heat, 0, None)               # keep positive contributions
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def kernel_shap_attribution(predict_fn, img_uint8: np.ndarray,
                            n_segments: int = 24, nsamples: int = 200) -> np.ndarray:
    """Optional KernelSHAP attribution over a simple grid segmentation.

    Provided for completeness; ``occlusion_attribution`` is the default because
    it is deterministic and cheaper. Requires the ``shap`` package.
    """
    import shap

    h, w = img_uint8.shape[:2]
    grid = int(np.ceil(np.sqrt(n_segments)))
    ch, cw = max(h // grid, 1), max(w // grid, 1)

    seg = np.zeros((h, w), dtype="int32")
    sid = 0
    regions: list[tuple[int, int, int, int]] = []
    for i in range(grid):
        for j in range(grid):
            y0, x0 = i * ch, j * cw
            y1 = h if i == grid - 1 else min((i + 1) * ch, h)
            x1 = w if j == grid - 1 else min((j + 1) * cw, w)
            if y0 >= y1 or x0 >= x1:
                continue
            seg[y0:y1, x0:x1] = sid
            regions.append((y0, y1, x0, x1))
            sid += 1

    mean_val = float(img_uint8.mean())

    def masked_predict(masks: np.ndarray) -> np.ndarray:
        out = []
        for m in masks:
            im = img_uint8.copy()
            for k, on in enumerate(m):
                if not on and k < len(regions):
                    y0, y1, x0, x1 = regions[k]
                    im[y0:y1, x0:x1] = mean_val
            out.append(float(predict_fn(im)))
        return np.array(out)

    explainer = shap.KernelExplainer(masked_predict, np.zeros((1, len(regions))))
    values = explainer.shap_values(np.ones((1, len(regions))), nsamples=nsamples,
                                   silent=True)
    vals = np.array(values).reshape(-1)

    heat = np.zeros((h, w), dtype="float64")
    for k, (y0, y1, x0, x1) in enumerate(regions):
        if k < vals.size:
            heat[y0:y1, x0:x1] = vals[k]
    heat = np.clip(heat, 0, None)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def _centroid(m: np.ndarray) -> tuple[float, float]:
    if m.sum() <= 0:
        idx = np.unravel_index(int(np.argmax(m)), m.shape)
        return float(idx[0]), float(idx[1])
    ys, xs = np.nonzero(m)
    wts = m[ys, xs]
    return float((ys * wts).sum() / wts.sum()), float((xs * wts).sum() / wts.sum())


def compare_maps(map_a: np.ndarray, map_b: np.ndarray,
                 top_percentile: float = 85.0) -> AgreementResult:
    """Quantify agreement between two attribution maps of the same shape."""
    import cv2

    if map_a.shape != map_b.shape:
        map_b = cv2.resize(map_b, (map_a.shape[1], map_a.shape[0]))

    a = map_a.astype("float64")
    b = map_b.astype("float64")
    for m in (a, b):
        if m.max() > 0:
            m /= m.max()

    # If either map is essentially flat there is no attribution signal to
    # compare - typically a confidently-negative prediction, where occluding any
    # region cannot lower an already-zero score. Calling that "agreement" would
    # inflate the agreement rate, so it is reported as inconclusive instead.
    degenerate = a.std() < 1e-9 or b.std() < 1e-9 or a.max() <= 0 or b.max() <= 0
    if degenerate:
        return AgreementResult(
            correlation=0.0, iou=0.0, peak_distance=1.0, agree=False,
            conclusive=False,
            verdict=("no attribution signal to compare (at least one method "
                     "produced a flat map, typical of a confidently negative "
                     "prediction) - agreement is undefined here"),
        )

    # Pearson correlation over all pixels.
    if a.std() < 1e-12 or b.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    # IoU of the top-attention regions.
    ta, tb = np.percentile(a, top_percentile), np.percentile(b, top_percentile)
    mask_a, mask_b = a >= max(ta, 1e-9), b >= max(tb, 1e-9)
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0

    # Distance between attention centroids, normalised by the image diagonal.
    ay, ax = _centroid(a * mask_a)
    by, bx = _centroid(b * mask_b)
    diag = float(np.hypot(*a.shape))
    peak_distance = float(np.hypot(ay - by, ax - bx) / diag) if diag else 1.0

    agree = (corr >= 0.3 and iou >= 0.20) or peak_distance <= 0.15
    if agree and corr >= 0.5:
        verdict = ("both methods localise the same region - explanation is "
                   "method-independent and therefore more trustworthy")
    elif agree:
        verdict = "methods broadly agree on the region of interest"
    else:
        verdict = ("explanation methods disagree on what drove the prediction - "
                   "treat the saliency map with caution and review manually")

    return AgreementResult(correlation=round(corr, 4), iou=round(iou, 4),
                           peak_distance=round(peak_distance, 4),
                           agree=bool(agree), verdict=verdict)


# --------------------------------------------------------------------------- #
# Convenience: build a predict_fn for any RadFrac model
# --------------------------------------------------------------------------- #
def make_predict_fn(model_key: str, predictor=None):
    """Return ``f(img_uint8) -> P(fractured)`` for a model key."""
    if predictor is None:
        from .inference import FracturePredictor
        predictor = FracturePredictor(model_key)

    def predict_fn(img_uint8: np.ndarray) -> float:
        return predictor.predict_array(img_uint8).probability

    return predict_fn
