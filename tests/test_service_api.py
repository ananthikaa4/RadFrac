"""Integration tests for the service layer (6.1) and HTTP API (6.2).

These train a tiny real model into a temp location so the whole stack is
exercised end-to-end without touching the project's real artifacts.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src import config


@pytest.fixture
def service_env(tiny_raw_dataset, tmp_path, monkeypatch):
    """Boot a service over the two ACTIVE models plus a frozen ensemble.

    Mirrors the production system: HOG+Canny+SVM and MobileNetV2. No CNN is
    trained or saved, so these tests also prove the app runs without it.
    """
    tf = pytest.importorskip("tensorflow")

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "svc.db")
    monkeypatch.setattr(config, "CASE_STORE_DIR", tmp_path / "cases")
    monkeypatch.setattr(config, "COST_THRESHOLDS_FILE", tmp_path / "cost.json")
    monkeypatch.setattr(config, "ENSEMBLE_CONFIG_FILE", tmp_path / "ensemble.json")
    monkeypatch.setattr(config, "MC_DROPOUT_PASSES", 5)   # keep tests fast

    from src.ensemble import EnsembleConfig
    from src.features_classical import build_feature_matrix
    from src.model_svm import build_classical, save_classical
    from src.model_transfer import build_transfer
    from src.preprocessing import (
        discover_samples,
        make_tf_dataset,
        save_splits,
        stratified_split,
    )
    from src.thresholds import save_threshold

    samples = discover_samples()
    train, val, test = stratified_split(samples)
    save_splits(train, val, test)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(0)

    xfer = build_transfer()
    xfer.fit(make_tf_dataset(train, training=True), epochs=1, verbose=0)
    xfer.save(config.TRANSFER_MODEL_FILE)
    save_threshold("transfer", 0.5)

    X, y = build_feature_matrix(train + val)
    pipe = build_classical("svm")
    pipe.fit(X, y)
    save_classical(pipe)
    save_threshold("classical", 0.5)

    EnsembleConfig(members=["classical", "transfer"],
                   weights={"classical": 0.5, "transfer": 0.5},
                   threshold=0.5, method="equal-average").save()

    from src.service import ScreeningService
    ScreeningService.reset()
    svc = ScreeningService.instance()
    yield svc, test
    ScreeningService.reset()


def _xray_like(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((140, 120, 3), 30, dtype="uint8")
    img[40:110, 40:80] = 200
    return np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype("uint8")


# --------------------------------------------------------------------------- #
# Service layer
# --------------------------------------------------------------------------- #
def test_service_loads_models_once(service_env):
    svc, _ = service_env
    assert set(svc.loaded_models) == {"classical", "transfer"}
    assert "cnn" not in svc.loaded_models          # archived, never active
    # Same object returned -> loaded once, not per call.
    assert svc._predictor("classical") is svc._predictor("classical")
    from src.service import ScreeningService
    assert ScreeningService.instance() is svc


def test_service_defaults_to_ensemble(service_env):
    svc, _ = service_env
    assert svc.ensemble_available() is True
    assert svc.default_model() == config.ENSEMBLE_KEY


def test_ensemble_runs_both_members_every_time(service_env):
    svc, _ = service_env
    res = svc.screen(_xray_like(11), persist=False)
    assert res.model_key == config.ENSEMBLE_KEY
    assert set(res.per_model) == {"classical", "transfer"}
    assert res.ensemble["used"] is True
    assert res.ensemble["gradcam_source"] == "transfer"
    for info in res.per_model.values():
        assert 0.0 <= info["probability"] <= 1.0
        assert info["label"] in config.CLASS_NAMES
        assert "display_name" in info


def test_agreement_reported_for_two_models(service_env):
    svc, _ = service_env
    res = svc.screen(_xray_like(12), persist=False)
    dis = res.disagreement
    assert dis["n_models"] == 2
    assert set(dis["labels"]) == {"classical", "transfer"}
    # Agreement is a boolean signal, and the reason is always populated.
    assert isinstance(dis["flagged"], bool)
    assert dis["reason"]


def test_service_screen_returns_full_result(service_env):
    svc, _ = service_env
    res = svc.screen(_xray_like(), model_key="transfer")
    assert 0.0 <= res.probability <= 1.0
    assert res.label in config.CLASS_NAMES
    assert res.prediction_id
    assert res.screening_note
    assert "not a diagnosis" in res.screening_note
    # Cross-check ran against the other loaded model.
    assert len(res.per_model) >= 2
    assert res.uncertainty.get("supported") is True
    assert res.uncertainty.get("passes") == 5


def test_service_rejects_invalid_image(service_env):
    from src.service import InvalidInputError

    svc, _ = service_env
    with pytest.raises(InvalidInputError) as exc:
        svc.screen(np.zeros((128, 128, 3), dtype="uint8"))
    assert exc.value.validity["errors"]


def test_service_persists_to_db(service_env):
    from src import db

    svc, _ = service_env
    res = svc.screen(_xray_like(1), model_key="transfer")
    rec = db.get_prediction(res.prediction_id)
    assert rec is not None
    assert rec["model_key"] == "transfer"
    assert rec["screening_note"]
    # Artifacts were written to the case store.
    from pathlib import Path
    assert Path(rec["image_path"]).exists()


def test_service_model_catalogue(service_env):
    svc, _ = service_env
    cat = svc.model_catalogue()
    keys = [m["key"] for m in cat["active_models"]]
    assert set(keys) == {"classical", "transfer"}
    assert "cnn" not in keys
    assert cat["archived_models"] == ["cnn"]
    assert cat["ensemble"] is not None
    assert cat["ensemble"]["name"] == "RADFRAC Ensemble"

    xfer = next(m for m in cat["active_models"] if m["key"] == "transfer")
    assert xfer["supports_uncertainty"] is True
    assert xfer["supports_gradcam"] is True
    svm = next(m for m in cat["active_models"] if m["key"] == "classical")
    # The SVM is not gradient-explainable; it gets its own feature view.
    assert svm["supports_gradcam"] is False
    assert "HOG" in svm["explainability"]


def test_referral_pdf_generates(service_env, tmp_path):
    from src.referral_report import build_referral_pdf

    svc, _ = service_env
    res = svc.screen(_xray_like(2))
    payload = res.to_dict()
    payload["patient"] = {"name": "Test Patient", "age": 41,
                          "region": "Wrist", "ref": "PHC-9"}
    pdf = build_referral_pdf(payload, tmp_path / "r.pdf")
    assert pdf.exists()
    assert pdf.read_bytes()[:5] == b"%PDF-"


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(service_env):
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_api_health_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert set(h["models_loaded"]) == {"classical", "transfer"}
    assert "cnn" not in h["models_loaded"]

    m = client.get("/models")
    assert m.status_code == 200
    body = m.json()
    assert [x["key"] for x in body["active_models"]] == ["classical", "transfer"]
    assert body["ensemble"]["key"] == config.ENSEMBLE_KEY

    e = client.get("/ensemble")
    assert e.status_code == 200
    assert set(e.json()["members"]) == {"classical", "transfer"}


def test_api_rejects_the_archived_cnn(client):
    """Asking for the archived model must fail, not silently run it."""
    r = client.post("/predict",
                    files={"file": ("x.png", _png_bytes(_xray_like(20)), "image/png")},
                    data={"model": "cnn"})
    assert r.status_code == 400
    assert "cnn" not in (r.json().get("detail") or "").split("available: ")[-1]


def test_api_predict_success(client):
    r = client.post("/predict",
                    files={"file": ("x.png", _png_bytes(_xray_like(3)), "image/png")})
    assert r.status_code == 200
    body = r.json()
    # Default path is the ensemble, and both members are reported.
    assert body["model_key"] == config.ENSEMBLE_KEY
    assert set(body["per_model"]) == {"classical", "transfer"}
    assert body["prediction_id"]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["screening_note"]


def test_api_rejects_invalid_image_with_422(client):
    black = _png_bytes(np.zeros((64, 64, 3), dtype="uint8"))
    r = client.post("/predict", files={"file": ("b.png", black, "image/png")})
    assert r.status_code == 422
    assert r.json()["rejected"] is True
    assert r.json()["validity"]["errors"]


def test_api_rejects_corrupt_upload(client):
    r = client.post("/predict", files={"file": ("x.png", b"garbage", "image/png")})
    assert r.status_code == 400


def test_api_rejects_unknown_model(client):
    r = client.post("/predict",
                    files={"file": ("x.png", _png_bytes(_xray_like(4)), "image/png")},
                    data={"model": "does_not_exist"})
    assert r.status_code == 400


def test_api_gradcam_and_report_endpoints(client):
    r = client.post("/predict",
                    files={"file": ("x.png", _png_bytes(_xray_like(5)), "image/png")},
                    data={"model": "transfer"})
    pid = r.json()["prediction_id"]

    g = client.get(f"/predict/{pid}/gradcam")
    assert g.status_code == 200
    assert g.headers["content-type"] == "image/png"

    rep = client.get(f"/predict/{pid}/report")
    assert rep.status_code == 200
    assert rep.content[:5] == b"%PDF-"


def test_api_404_for_unknown_prediction(client):
    assert client.get("/predict/nonexistent").status_code == 404
    assert client.get("/predict/nonexistent/gradcam").status_code == 404


def test_api_batch_returns_triage_order(client):
    files = [("files", (f"x{i}.png", _png_bytes(_xray_like(i + 6)), "image/png"))
             for i in range(3)]
    r = client.post("/predict/batch", files=files, data={"with_uncertainty": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["triage_order"]) == 3


def test_api_case_flow(client):
    case = client.post("/cases", json={"patient_ref": "P1", "body_region": "wrist"})
    assert case.status_code == 201
    cid = case.json()["case_id"]
    client.post("/predict",
                files={"file": ("x.png", _png_bytes(_xray_like(9)), "image/png")},
                data={"case_id": cid})
    got = client.get(f"/cases/{cid}").json()
    assert got["patient_ref"] == "P1"
    assert len(got["predictions"]) == 1


def test_api_triage_endpoint(client):
    client.post("/predict",
                files={"file": ("x.png", _png_bytes(_xray_like(10)), "image/png")})
    t = client.get("/triage").json()
    assert "queue" in t
    assert "does not replace sequential review" in t["note"]
