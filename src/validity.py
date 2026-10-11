"""Input validity gate - "is this plausibly an X-ray?" (4.8).

Guards against garbage-in-confident-garbage-out. A classifier will happily emit
"87% fractured" for a photo of a car park, which is actively dangerous in a
screening context where the operator may not be a specialist. This module runs
cheap, explainable checks BEFORE inference and returns structured reasons.

Checks (all thresholds env-overridable via ``src.config``):

* **Too dark / too bright** - a blank or blown-out capture carries no signal.
* **Low contrast** - near-uniform images (lens cap, blank sheet, solid colour).
* **Implausible aspect ratio** - radiographs are roughly squarish; a 16:9
  screenshot or a panorama is suspect.
* **Too colourful** - radiographs are greyscale; strong per-pixel channel
  separation means a colour photograph.

Deliberately a *warning* gate, not a hard block, for most checks: the service
layer decides policy. Refusing a borderline-dark but genuine film would be worse
than flagging it, so severity is reported and the caller chooses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import config


@dataclass
class ValidityResult:
    ok: bool                              # passed every hard check
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def should_block(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.errors:
            return "rejected: " + "; ".join(self.errors)
        if self.warnings:
            return "accepted with warnings: " + "; ".join(self.warnings)
        return "input checks passed"


def _grayscale_saturation(img_uint8: np.ndarray) -> float:
    """Mean per-pixel channel spread, normalised to [0, 1].

    0 for a true greyscale image (R==G==B); rises with colour content.
    """
    if img_uint8.ndim == 2 or img_uint8.shape[-1] == 1:
        return 0.0
    arr = img_uint8.astype("float32")
    spread = arr.max(axis=-1) - arr.min(axis=-1)
    return float(spread.mean() / 255.0)


def check_image(img_uint8: np.ndarray) -> ValidityResult:
    """Run all validity checks on an RGB/greyscale uint8 image array."""
    res = ValidityResult(ok=True)

    if img_uint8.size == 0:
        res.ok = False
        res.errors.append("image is empty")
        return res

    gray = (img_uint8.astype("float32").mean(axis=-1) / 255.0
            if img_uint8.ndim == 3 else img_uint8.astype("float32") / 255.0)

    mean_i = float(gray.mean())
    std_i = float(gray.std())
    h, w = gray.shape[:2]
    aspect = max(h, w) / max(1.0, min(h, w))
    saturation = _grayscale_saturation(img_uint8)

    res.metrics = {
        "mean_intensity": round(mean_i, 4),
        "intensity_std": round(std_i, 4),
        "height": int(h),
        "width": int(w),
        "aspect_ratio": round(aspect, 3),
        "color_saturation": round(saturation, 4),
    }

    # --- hard errors: no usable signal at all -------------------------------
    if mean_i < config.MIN_MEAN_INTENSITY:
        res.errors.append(
            f"image is almost entirely black (mean intensity {mean_i:.3f} < "
            f"{config.MIN_MEAN_INTENSITY})")
    if mean_i > config.MAX_MEAN_INTENSITY:
        res.errors.append(
            f"image is almost entirely white (mean intensity {mean_i:.3f} > "
            f"{config.MAX_MEAN_INTENSITY})")
    if std_i < config.MIN_INTENSITY_STD:
        res.errors.append(
            f"image has almost no contrast (sd {std_i:.4f} < "
            f"{config.MIN_INTENSITY_STD}); not a diagnostic image")

    # --- warnings: suspicious but possibly genuine --------------------------
    if saturation > config.MAX_COLOR_SATURATION:
        res.warnings.append(
            f"image looks like a colour photograph rather than a radiograph "
            f"(colour saturation {saturation:.3f} > {config.MAX_COLOR_SATURATION})")
    if aspect > config.MAX_ASPECT_RATIO:
        res.warnings.append(
            f"unusual aspect ratio {aspect:.2f}:1 for a radiograph")
    if min(h, w) < 64:
        res.warnings.append(
            f"low resolution ({w}x{h}); prediction reliability is reduced")

    res.ok = not res.errors
    return res


def check_gradcam_coherence(heatmap: np.ndarray,
                            min_focus_ratio: float = 0.002,
                            max_focus_ratio: float = 0.85) -> ValidityResult:
    """Post-hoc sanity check on model attention.

    If Grad-CAM attends to essentially nothing, or floods the entire frame, the
    model is not locking onto anatomical structure and the prediction deserves a
    manual-review flag. Complements the pre-inference pixel checks.
    """
    res = ValidityResult(ok=True)
    if heatmap is None or heatmap.size == 0:
        res.warnings.append("no attention map produced")
        return res

    hot = float((heatmap >= 0.5).mean())
    res.metrics = {"attention_area_ratio": round(hot, 4)}
    if hot < min_focus_ratio:
        res.warnings.append(
            "model attention is diffuse/empty - no clear anatomical focus; "
            "recommend manual review")
    elif hot > max_focus_ratio:
        res.warnings.append(
            "model attention covers nearly the whole image - not localised to a "
            "structure; recommend manual review")
    return res
