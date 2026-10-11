"""Tests for the two-model active system: ensemble, agreement, CNN exclusion."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from src import config


# --------------------------------------------------------------------------- #
# CNN must be archived, not active
# --------------------------------------------------------------------------- #
def test_active_keys_exclude_cnn():
    assert "cnn" not in config.ACTIVE_MODEL_KEYS
    assert "cnn" in config.ARCHIVED_MODEL_KEYS
    assert set(config.ACTIVE_MODEL_KEYS) == {"classical", "transfer"}


def test_available_models_hides_cnn(tmp_path, monkeypatch):
    from src import inference

    # Patching config alone must be enough - artifact paths are resolved on each
    # call, not frozen at import.
    for name in ("CNN_MODEL_FILE", "TRANSFER_MODEL_FILE", "CLASSICAL_MODEL_FILE"):
        p = tmp_path / f"{name}.bin"
        p.write_bytes(b"x")
        monkeypatch.setattr(config, name, p)

    assert "cnn" not in inference.available_models()
    assert set(inference.available_models()) == {"classical", "transfer"}
    # Archived models are still reachable for documentation/training.
    assert "cnn" in inference.available_models(include_archived=True)


# --------------------------------------------------------------------------- #
# Calibration + ensemble maths
# --------------------------------------------------------------------------- #
def test_platt_calibrator_preserves_ranking_and_improves_scale():
    from src.ensemble import PlattCalibrator

    rng = np.random.default_rng(0)
    # Over-confident scores: true rate is much closer to 0.5 than the score says.
    y = rng.integers(0, 2, 600)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, 600), rng.uniform(0.0, 0.4, 600))
    squashed = 0.5 + (p - 0.5) * 1.8
    squashed = np.clip(squashed, 0.001, 0.999)

    cal = PlattCalibrator().fit(squashed, y)
    out = cal.transform(squashed)
    # Platt scaling is monotonic, so it must leave ranking - and therefore
    # ROC-AUC - untouched, changing only the probability scale. (Asserted via
    # AUC rather than argsort, because clipping creates ties whose argsort order
    # is arbitrary.)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, out) == pytest.approx(roc_auc_score(y, squashed),
                                                  abs=1e-9)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # And it must actually move the scale (coefficient fitted, not identity).
    assert cal.fitted and abs(cal.coef - 1.0) > 1e-6


def test_ensemble_combine_is_weighted_average_of_calibrated_members():
    from src.ensemble import EnsembleConfig

    cfg = EnsembleConfig(
        members=["classical", "transfer"],
        weights={"classical": 0.5, "transfer": 0.5},
        calibrators={},          # identity calibration
        threshold=0.4,
    )
    out = cfg.combine({"classical": np.array([0.8]), "transfer": np.array([0.2])})
    assert out[0] == pytest.approx(0.5, abs=1e-6)

    cfg.weights = {"classical": 0.75, "transfer": 0.25}
    out = cfg.combine({"classical": np.array([1.0]), "transfer": np.array([0.0])})
    assert out[0] == pytest.approx(0.75, abs=1e-6)


def test_ensemble_renormalises_when_a_member_is_missing():
    from src.ensemble import EnsembleConfig

    cfg = EnsembleConfig(members=["classical", "transfer"],
                         weights={"classical": 0.5, "transfer": 0.5})
    out = cfg.combine({"classical": np.array([0.9])})
    assert out[0] == pytest.approx(0.9, abs=1e-6)


def test_ensemble_config_roundtrip(tmp_path, monkeypatch):
    from src.ensemble import EnsembleConfig

    path = tmp_path / "ens.json"
    monkeypatch.setattr(config, "ENSEMBLE_CONFIG_FILE", path)
    cfg = EnsembleConfig(weights={"classical": 0.4, "transfer": 0.6},
                         threshold=0.37, method="weighted-average")
    cfg.save()
    back = EnsembleConfig.load()
    assert back is not None
    assert back.threshold == pytest.approx(0.37)
    assert back.weights["transfer"] == pytest.approx(0.6)


def test_hard_vote_and_or_semantics():
    from src.ensemble import score_hard

    y = np.array([1, 1, 0, 0])
    preds = {"a": np.array([1, 0, 1, 0]), "b": np.array([1, 1, 0, 0])}
    m_and = score_hard(y, preds, "and")
    m_or = score_hard(y, preds, "or")
    # AND fires only where both agree positive -> 1 TP
    assert m_and["tp"] == 1
    # OR fires where either is positive -> 2 TP but picks up a FP
    assert m_or["tp"] == 2
    assert m_or["fp"] == 1


def test_score_reports_confusion_and_rates():
    from src.ensemble import score

    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    m = score(y, p, 0.5)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["fpr"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Classical explainability
# --------------------------------------------------------------------------- #
def test_classical_feature_visualisation(tmp_path):
    from src.explain_classical import (
        edge_density_grid,
        hog_visualisation,
        render_classical_explanation,
    )

    rng = np.random.default_rng(3)
    img = np.full((*config.IMAGE_SIZE, 3), 40, dtype="uint8")
    img[40:90, 50:78] = 210
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype("uint8")

    hog_img, edges = hog_visualisation(img)
    assert hog_img.shape == config.IMAGE_SIZE
    assert edges.shape == config.IMAGE_SIZE
    grid = edge_density_grid(edges)
    assert grid.shape == (4, 4)
    assert np.all((grid >= 0) & (grid <= 1))

    out = render_classical_explanation(img, tmp_path / "cls.png")
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# PDF with patient information
# --------------------------------------------------------------------------- #
def _result_payload(**over) -> dict:
    payload = {
        "prediction_id": "testcase01",
        "model_key": "ensemble",
        "probability": 0.784,
        "threshold": 0.33,
        "label": config.POSITIVE_CLASS,
        "positive": True,
        "recommend_manual_review": True,
        "patient": {"name": "A. Patient", "age": 34,
                    "region": "Wrist", "ref": "PHC-1"},
        "per_model": {
            "classical": {"probability": 0.842, "label": config.POSITIVE_CLASS,
                          "threshold": 0.295,
                          "display_name": "HOG + Canny + SVM"},
            "transfer": {"probability": 0.790, "label": config.POSITIVE_CLASS,
                         "threshold": 0.285,
                         "display_name": "MobileNetV2 Transfer Learning"},
        },
        "ensemble": {"used": True, "method": "equal-average",
                     "weights": {"classical": 0.5, "transfer": 0.5}},
        "uncertainty": {"supported": True, "std": 0.031, "passes": 40},
        "validity": {"ok": True, "warnings": []},
        "screening_note": "Screening note text.",
        "synthetic_data": False,
    }
    payload.update(over)
    return payload


def test_pdf_contains_patient_and_both_models(tmp_path):
    from src.referral_report import build_referral_pdf

    out = build_referral_pdf(_result_payload(), tmp_path / "r.pdf")
    assert out.exists()
    raw = out.read_bytes()
    assert raw[:5] == b"%PDF-"
    assert len(raw) > 3000


def test_pdf_handles_disagreement_and_missing_images(tmp_path):
    from src.referral_report import build_referral_pdf

    payload = _result_payload()
    payload["per_model"]["transfer"]["label"] = config.CLASS_NAMES[
        1 - config.POSITIVE_INDEX]
    payload["image_path"] = str(tmp_path / "missing.png")   # absent on purpose
    out = build_referral_pdf(payload, tmp_path / "r2.pdf")
    assert out.exists() and out.read_bytes()[:5] == b"%PDF-"


def test_pdf_never_mentions_the_archived_cnn(tmp_path):
    """The report must not surface an archived model to the reader."""
    from src.referral_report import _member_rows

    payload = _result_payload()
    payload["per_model"]["cnn"] = {"probability": 0.5, "label": config.POSITIVE_CLASS,
                                   "threshold": 0.5, "display_name": "Custom CNN"}
    rows = _member_rows(payload)
    assert [k for k, _ in rows] == ["classical", "transfer"]
    assert all(k != "cnn" for k, _ in rows)


# --------------------------------------------------------------------------- #
# UI module import + theme
# --------------------------------------------------------------------------- #
def test_theme_css_has_no_sidebar_and_defines_tokens():
    from app.theme import COLORS, css

    sheet = css()
    assert "stSidebar" in sheet and "display: none" in sheet
    for key in ("bg", "surface", "accent", "positive", "negative"):
        assert COLORS[key].startswith("#")


def test_demo_app_imports_and_excludes_cnn():
    import app.demo_app as ui

    src = (ui.__file__)
    text = open(src, encoding="utf-8").read()
    # No user-facing CNN anywhere in the UI.
    assert "Custom CNN" not in text.replace(
        "A custom CNN was evaluated during development and is archived; ", "")
    # Body region is no longer collected: it never fed either model, so the UI
    # must not present it as a field or echo it back on the result page.
    assert "BODY_REGIONS" not in text
    assert "Body region" not in text
    # The screen is not branded as AI-assisted.
    assert "AI-assisted" not in text
    assert hasattr(ui, "stage_upload") and hasattr(ui, "stage_result")
