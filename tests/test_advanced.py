"""Tests for uncertainty, disagreement, screening notes, thresholds, explainability."""
from __future__ import annotations

import numpy as np
import pytest

from src import config


# --------------------------------------------------------------------------- #
# MC Dropout uncertainty (4.1)
# --------------------------------------------------------------------------- #
def test_mc_dropout_produces_variance_and_keeps_bn_in_inference():
    tf = pytest.importorskip("tensorflow")
    from src.model_cnn import build_cnn
    from src.uncertainty import mc_dropout_predict, model_has_dropout

    model = build_cnn()
    assert model_has_dropout(model) is True

    img = np.random.rand(*config.IMAGE_SIZE, config.IMAGE_CHANNELS).astype("float32")
    res = mc_dropout_predict(model, img, passes=12)
    assert res.supported is True
    assert res.passes == 12
    assert 0.0 <= res.mean <= 1.0
    assert res.std >= 0.0
    assert res.lower <= res.upper

    # Deterministic inference must be unchanged by the MC clone (BatchNorm still
    # in inference mode), so a plain predict is reproducible.
    a = float(model.predict(img[None], verbose=0).reshape(-1)[0])
    b = float(model.predict(img[None], verbose=0).reshape(-1)[0])
    assert a == pytest.approx(b, abs=1e-6)


def test_mc_dropout_reports_unsupported_without_dropout():
    tf = pytest.importorskip("tensorflow")
    from src.uncertainty import mc_dropout_predict

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    res = mc_dropout_predict(model, np.zeros((*config.IMAGE_SIZE, 3), "float32"))
    assert res.supported is False
    assert "Dropout" in res.reason


# --------------------------------------------------------------------------- #
# Disagreement (4.7)
# --------------------------------------------------------------------------- #
def test_disagreement_flags_label_split():
    from src.disagreement import assess

    r = assess({
        "cnn": {"probability": 0.8, "label": "fractured"},
        "transfer": {"probability": 0.2, "label": "not_fractured"},
    })
    assert r.flagged is True
    assert "disagree" in r.reason


def test_disagreement_flags_wide_spread_despite_same_label():
    from src.disagreement import assess

    r = assess({
        "cnn": {"probability": 0.55, "label": "fractured"},
        "classical": {"probability": 0.99, "label": "fractured"},
    })
    assert r.flagged is True
    assert "weak consensus" in r.reason


def test_disagreement_clean_consensus():
    from src.disagreement import assess

    r = assess({
        "cnn": {"probability": 0.91, "label": "fractured"},
        "transfer": {"probability": 0.88, "label": "fractured"},
    })
    assert r.flagged is False
    assert r.majority_label == "fractured"


def test_disagreement_single_model_cannot_crosscheck():
    from src.disagreement import assess

    r = assess({"cnn": {"probability": 0.7, "label": "fractured"}})
    assert r.flagged is False
    assert "one model" in r.reason


# --------------------------------------------------------------------------- #
# Screening note (4.9)
# --------------------------------------------------------------------------- #
def test_note_mentions_location_and_requires_signoff():
    from src.screening_note import describe_attention, generate_note

    heat = np.zeros((64, 64), dtype="float32")
    heat[40:50, 28:36] = 1.0            # lower third, central
    att = describe_attention(heat)
    assert att["available"] is True
    assert "distal" in att["vertical"]

    note = generate_note(
        label="fractured", probability=0.88, threshold=0.4, model_key="cnn",
        uncertainty={"supported": True, "std": 0.02, "flagged": False, "passes": 40},
        disagreement={"n_models": 3, "flagged": False, "labels": {
            "cnn": "fractured", "transfer": "fractured", "classical": "fractured"}},
        attention=att, validity={"warnings": []}, synthetic=False,
    )
    assert "distal" in note
    assert "sign-off" in note or "signed" in note or "qualified clinician" in note
    assert "not a diagnosis" in note


def test_note_flags_synthetic_and_uncertainty():
    from src.screening_note import generate_note

    note = generate_note(
        label="fractured", probability=0.55, threshold=0.5, model_key="cnn",
        uncertainty={"supported": True, "std": 0.4, "flagged": True, "passes": 40},
        disagreement={"n_models": 1, "flagged": False},
        attention={"available": False}, validity={"warnings": []}, synthetic=True,
    )
    assert "SYNTHETIC" in note
    assert "not reliable" in note


def test_note_negative_case_advises_clinical_override():
    from src.screening_note import generate_note

    note = generate_note(
        label="not_fractured", probability=0.05, threshold=0.4, model_key="cnn",
        uncertainty={"supported": True, "std": 0.01, "flagged": False, "passes": 40},
        disagreement={"n_models": 3, "flagged": False, "labels": {}},
        attention={"available": False}, validity={"warnings": []}, synthetic=False,
    )
    assert "does not exclude" in note


# --------------------------------------------------------------------------- #
# Cost-sensitive thresholds + calibration (section 3, 4.4)
# --------------------------------------------------------------------------- #
def test_cost_sweep_prefers_recall_when_fn_expensive():
    from src.threshold_analysis import sweep_costs

    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    # Overlapping scores so the threshold genuinely matters.
    p = np.concatenate([rng.normal(0.35, 0.15, 50), rng.normal(0.65, 0.15, 50)])
    p = np.clip(p, 0, 1)

    cheap_fn = sweep_costs(y, p, fn_cost=1.0, fp_cost=1.0, recall_target=None)
    costly_fn = sweep_costs(y, p, fn_cost=20.0, fp_cost=1.0, recall_target=None)

    # Making false negatives expensive must not reduce recall.
    assert costly_fn.at_best["recall"] >= cheap_fn.at_best["recall"]
    assert costly_fn.best_threshold <= cheap_fn.best_threshold


def test_cost_sweep_reports_threshold_for_recall_target():
    from src.threshold_analysis import sweep_costs

    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.45, 0.55, 0.8, 0.9])
    a = sweep_costs(y, p, recall_target=1.0)
    assert a.threshold_for_recall is not None
    # At that threshold every positive is caught.
    assert all(pp >= a.threshold_for_recall for pp in p[y == 1])


def test_calibration_detects_well_and_poorly_calibrated():
    from src.threshold_analysis import calibration

    rng = np.random.default_rng(1)
    # Well calibrated: label drawn with probability equal to the score.
    p_good = rng.uniform(0, 1, 4000)
    y_good = (rng.uniform(0, 1, 4000) < p_good).astype(int)
    good = calibration(y_good, p_good)
    assert good.ece < 0.05

    # Poorly calibrated: confident scores that are frequently wrong.
    p_bad = np.full(400, 0.95)
    y_bad = np.zeros(400, dtype=int)
    bad = calibration(y_bad, p_bad)
    assert bad.ece > 0.5
    assert "poorly calibrated" in bad.verdict


def test_threshold_source_prefers_cost_then_f1(tmp_path, monkeypatch):
    import json

    from src import thresholds as th

    monkeypatch.setattr(config, "THRESHOLDS_FILE", tmp_path / "f1.json")
    monkeypatch.setattr(config, "COST_THRESHOLDS_FILE", tmp_path / "cost.json")

    assert th.get_threshold_with_source("cnn") == (config.DEFAULT_THRESHOLD, "default")

    config.THRESHOLDS_FILE.write_text(json.dumps({"cnn": 0.42}))
    assert th.get_threshold_with_source("cnn") == (0.42, "f1")

    config.COST_THRESHOLDS_FILE.write_text(json.dumps({"cnn": 0.11}))
    assert th.get_threshold_with_source("cnn") == (0.11, "cost")


# --------------------------------------------------------------------------- #
# Dual explainability agreement (4.5)
# --------------------------------------------------------------------------- #
def test_compare_maps_agrees_on_same_region():
    from src.explain_dual import compare_maps

    a = np.zeros((64, 64), dtype="float32")
    a[20:40, 20:40] = 1.0
    b = np.zeros((64, 64), dtype="float32")
    b[22:42, 22:42] = 1.0
    r = compare_maps(a, b)
    assert r.conclusive is True
    assert r.agree is True
    assert r.iou > 0.2


def test_compare_maps_disagrees_on_opposite_regions():
    from src.explain_dual import compare_maps

    a = np.zeros((64, 64), dtype="float32")
    a[0:16, 0:16] = 1.0
    b = np.zeros((64, 64), dtype="float32")
    b[48:64, 48:64] = 1.0
    r = compare_maps(a, b)
    assert r.conclusive is True
    assert r.agree is False


def test_compare_maps_reports_inconclusive_for_flat_map():
    """Regression guard: a flat map must NOT be counted as agreement."""
    from src.explain_dual import compare_maps

    a = np.zeros((32, 32), dtype="float32")
    b = np.zeros((32, 32), dtype="float32")
    b[10:20, 10:20] = 1.0
    r = compare_maps(a, b)
    assert r.conclusive is False
    assert r.agree is False
    assert "no attribution signal" in r.verdict


def test_occlusion_attribution_finds_the_decisive_region():
    from src.explain_dual import occlusion_attribution

    img = np.full((64, 64, 3), 40, dtype="uint8")
    img[28:36, 28:36] = 220              # the "lesion"

    def predict_fn(im: np.ndarray) -> float:
        # Score depends only on brightness of the central patch.
        return float(im[28:36, 28:36].mean() / 255.0)

    heat = occlusion_attribution(predict_fn, img, patch=16, stride=8)
    assert heat.shape == (64, 64)
    # Attention must peak on the decisive central region, not the corners.
    assert heat[28:36, 28:36].mean() > heat[0:8, 0:8].mean()


# --------------------------------------------------------------------------- #
# Distillation (4.3)
# --------------------------------------------------------------------------- #
def test_student_is_much_smaller_than_teacher():
    pytest.importorskip("tensorflow")
    from src.distillation import build_student, model_footprint
    from src.model_cnn import build_cnn

    teacher = build_cnn()
    student = build_student(width=0.5)
    assert (model_footprint(student)["parameters"]
            < model_footprint(teacher)["parameters"])


def test_distillation_loss_rewards_matching_the_teacher():
    tf = pytest.importorskip("tensorflow")
    from src.distillation import make_distillation_loss

    loss_fn = make_distillation_loss(alpha=0.7, temperature=4.0)
    # y_true packs [hard_label, teacher_probability].
    y_true = tf.constant([[1.0, 0.9]])
    close = float(tf.reduce_mean(loss_fn(y_true, tf.constant([[0.88]]))))
    far = float(tf.reduce_mean(loss_fn(y_true, tf.constant([[0.10]]))))
    assert close < far
