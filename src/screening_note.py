"""Template-based natural-language screening note generator (4.9).

Turns raw model output into a short, readable note a referring physician can
actually use. Deliberately **template-based, not an LLM**: every clause is
derived from a measured quantity, so the note cannot hallucinate a finding that
the pipeline did not compute. That property matters more in a clinical screening
context than fluent prose.

What each clause is grounded in:

* location wording  -> centroid of the Grad-CAM attention mass (which third of
  the image, and whether central or peripheral)
* extent wording    -> area of the thresholded attention region
* confidence wording-> the calibrated probability vs the operating threshold
* stability wording -> Monte Carlo Dropout standard deviation
* corroboration     -> the other models' verdicts (disagreement module)

The note always ends with an explicit human-sign-off requirement; RadFrac is a
screening aid, never a diagnostic authority.
"""
from __future__ import annotations

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Attention-derived morphology descriptors
# --------------------------------------------------------------------------- #
def describe_attention(heatmap: np.ndarray | None, threshold: float = 0.5) -> dict:
    """Summarise where and how broadly the model is attending."""
    if heatmap is None or heatmap.size == 0:
        return {"available": False}

    mask = heatmap >= threshold
    area_ratio = float(mask.mean())
    if not mask.any():
        # Fall back to the single hottest point.
        idx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        cy, cx = float(idx[0]), float(idx[1])
    else:
        ys, xs = np.nonzero(mask)
        cy, cx = float(ys.mean()), float(xs.mean())

    h, w = heatmap.shape[:2]
    y_frac, x_frac = cy / max(h - 1, 1), cx / max(w - 1, 1)

    vertical = ("proximal (upper) third" if y_frac < 0.34
                else "mid-shaft region" if y_frac < 0.67
                else "distal (lower) third")
    horizontal = ("medial/left aspect" if x_frac < 0.34
                  else "central axis" if x_frac < 0.67
                  else "lateral/right aspect")
    extent = ("a focal area" if area_ratio < 0.08
              else "a moderately sized area" if area_ratio < 0.30
              else "a broad area")

    return {
        "available": True,
        "vertical": vertical,
        "horizontal": horizontal,
        "extent": extent,
        "area_ratio": round(area_ratio, 4),
        "centroid_yx": [round(y_frac, 3), round(x_frac, 3)],
    }


def _confidence_word(prob: float, threshold: float) -> str:
    """Distance from the operating threshold, expressed in words."""
    margin = abs(prob - threshold)
    if margin >= 0.35:
        return "high"
    if margin >= 0.15:
        return "moderate"
    return "borderline"


# --------------------------------------------------------------------------- #
# Note assembly
# --------------------------------------------------------------------------- #
def generate_note(
    *,
    label: str,
    probability: float,
    threshold: float,
    model_key: str,
    uncertainty: dict | None = None,
    disagreement: dict | None = None,
    attention: dict | None = None,
    validity: dict | None = None,
    synthetic: bool = False,
) -> str:
    """Compose the screening note from computed pipeline outputs."""
    positive = label == config.POSITIVE_CLASS
    conf = _confidence_word(probability, threshold)
    sentences: list[str] = []

    # 1. Primary finding, with attention-derived localisation when available.
    if positive:
        if attention and attention.get("available"):
            sentences.append(
                f"Screening flags a possible fracture: model attention "
                f"concentrates on {attention['extent']} in the "
                f"{attention['vertical']}, toward the {attention['horizontal']}."
            )
        else:
            sentences.append("Screening flags a possible fracture.")
    else:
        sentences.append(
            "No fracture features detected above the screening threshold on this image."
        )

    # 2. Quantified confidence relative to the operating point.
    sentences.append(
        f"Fracture probability {probability:.0%} against an operating threshold of "
        f"{threshold:.0%} ({conf} margin)."
    )

    # 3. Stability from MC Dropout.
    if uncertainty and uncertainty.get("supported"):
        std = uncertainty.get("std")
        if uncertainty.get("flagged"):
            sentences.append(
                f"Repeat sampling shows unstable output (sd={std:.3f} over "
                f"{uncertainty.get('passes')} passes), so this prediction is "
                f"not reliable on its own."
            )
        else:
            sentences.append(
                f"Repeat sampling is consistent (sd={std:.3f}), indicating a "
                f"stable model response."
            )

    # 4. Cross-model corroboration.
    if disagreement and disagreement.get("n_models", 0) >= 2:
        if disagreement.get("flagged"):
            sentences.append(f"Cross-check: {disagreement['reason']}.")
        else:
            others = [k for k in disagreement.get("labels", {}) if k != model_key]
            if others:
                sentences.append(
                    f"Cross-check: independent model(s) "
                    f"({', '.join(sorted(others))}) concur."
                )

    # 5. Input-quality caveats.
    if validity:
        for w in validity.get("warnings", [])[:2]:
            sentences.append(f"Image-quality note: {w}.")

    # 6. Action framing for a non-specialist at the point of care.
    if positive or (uncertainty or {}).get("flagged") or (disagreement or {}).get("flagged"):
        sentences.append(
            "Recommend review by a qualified radiologist and onward referral "
            "per local protocol."
        )
    else:
        sentences.append(
            "If clinical suspicion persists despite this negative screen, refer "
            "regardless - a negative screen does not exclude fracture."
        )

    # 7. Mandatory framing + synthetic-data guard.
    sentences.append(
        f"Generated by RadFrac ({model_key}) as a screening aid; it is not a "
        f"diagnosis and requires sign-off by a qualified clinician."
    )
    if synthetic:
        sentences.append(
            "WARNING: this model was trained on SYNTHETIC data and must not be "
            "used for any clinical purpose."
        )

    return " ".join(sentences)
