"""Tests for the gradient-boosting extension (no TensorFlow required)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src import config
from src.features_hybrid import (DEEP_DIM, HANDCRAFTED_DIM, assemble, family_slices,
                                 hybrid_vector, resolve_path)
from src.model_boosting import (BOOSTER_KEYS, COMMITTEE_KEY, BoostingSuite, assemble_pipeline,
                                build_classifier, build_preprocessor, load_yaml_config,
                                positive_proba, save_model, save_registry)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n, d = 240, 40
    # Correlated signal block so it survives scaling + PCA (PCA keeps the
    # dominant shared-variance directions).
    z = rng.normal(size=n)
    X = rng.normal(size=(n, d)).astype("float32")
    X[:, :6] += 3 * z[:, None]
    y = (z + rng.normal(scale=.4, size=n) > 0).astype(int)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture
def boost_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path / "models" / "boosting"


def test_yaml_config_has_all_search_spaces():
    cfg = load_yaml_config()
    for key in BOOSTER_KEYS:
        assert key in cfg["search_spaces"]
        assert key in cfg["tuning"]["n_trials"]
    assert cfg["operating_point"]["max_fpr"] == pytest.approx(config.MAX_FALSE_POSITIVE_RATE)


def test_family_slices_cover_hybrid_vector():
    s = family_slices()
    assert s["hog"].stop == s["edges"].start
    assert s["edges"].stop == HANDCRAFTED_DIM
    assert s["deep"].stop == HANDCRAFTED_DIM + DEEP_DIM
    assert list(family_slices(use_handcrafted=False))[0] == "deep"


def test_handcrafted_vector_dimension():
    img = np.random.default_rng(1).integers(0, 255, (300, 220, 3), dtype=np.uint8)
    v = hybrid_vector(img, use_deep=False)
    assert v.shape == (HANDCRAFTED_DIM,)
    with pytest.raises(ValueError):
        hybrid_vector(img, embedding_model=None, use_deep=True)


def test_resolve_path_remaps_foreign_absolute_paths(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "fractured").mkdir(parents=True)
    (raw / "fractured" / "xr_000001.jpg").write_bytes(b"x")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    got = resolve_path(r"D:\Old Laptop\radfrac\data\raw\fractured\xr_000001.jpg")
    assert got == str(raw / "fractured" / "xr_000001.jpg")


def test_assemble_concatenates_blocks():
    data = {"handcrafted": np.ones((3, 4)), "deep": np.zeros((3, 2))}
    assert assemble(data).shape == (3, 6)
    assert assemble(data, use_deep=False).shape == (3, 4)


@pytest.mark.parametrize("key", BOOSTER_KEYS)
def test_each_booster_trains_and_ranks(key, toy_data):
    from sklearn.metrics import roc_auc_score

    from scripts.train_boosting import fit_with_early_stopping, learning_curves, fit_final

    Xtr, ytr, Xv, yv = toy_data
    pre = build_preprocessor(8).fit(Xtr)
    Ztr, Zv = pre.transform(Xtr), pre.transform(Xv)
    params = {"learning_rate": 0.1}
    clf, best_it = fit_with_early_stopping(key, params, 200, 20, Ztr, ytr, Zv, yv, 0)
    assert best_it >= 1
    final = fit_final(key, params, best_it, Ztr, ytr, Zv, yv, 0)
    curves = learning_curves(key, final)
    assert len(curves["train"]) == len(curves["val"]) > 0
    pipe = assemble_pipeline(pre, final)
    p = positive_proba(pipe, Xv)
    assert p.shape == (len(yv),) and np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(yv, p) > 0.7


def test_unknown_booster_rejected():
    with pytest.raises(ValueError):
        build_classifier("adaboost", {}, 10)


def test_suite_load_and_committee(boost_dir, toy_data):
    Xtr, ytr, Xv, _ = toy_data
    pre = build_preprocessor(8).fit(Xtr)
    models = {}
    for key in BOOSTER_KEYS:
        clf = build_classifier(key, {"learning_rate": 0.1}, 30, 0).fit(pre.transform(Xtr), ytr)
        models[key] = save_model(key, assemble_pipeline(pre, clf))
    save_registry({
        "features": {"use_handcrafted": True, "use_deep": False, "dim": Xtr.shape[1]},
        "models": {**{k: {"threshold": 0.5, "test_metrics": {"roc_auc": 0.9}} for k in BOOSTER_KEYS},
                   COMMITTEE_KEY: {"threshold": 0.4, "test_metrics": {"roc_auc": 0.91}}},
    })
    suite = BoostingSuite.load()
    assert suite is not None and not suite.needs_deep
    assert suite.keys == list(BOOSTER_KEYS) + [COMMITTEE_KEY]
    out = suite.predict_vector(Xv[0])
    mean = np.mean([out[k]["probability"] for k in BOOSTER_KEYS])
    assert out[COMMITTEE_KEY]["probability"] == pytest.approx(mean)
    for v in out.values():
        assert v["label"] in config.CLASS_NAMES
        assert (v["label"] == config.POSITIVE_CLASS) == (v["probability"] >= v["threshold"])


def test_suite_absent_without_registry(boost_dir):
    assert BoostingSuite.load() is None


def test_suite_skips_corrupt_artifact(boost_dir, toy_data):
    Xtr, ytr, _, _ = toy_data
    pre = build_preprocessor(8).fit(Xtr)
    clf = build_classifier("xgboost", {}, 20, 0).fit(pre.transform(Xtr), ytr)
    save_model("xgboost", assemble_pipeline(pre, clf))
    boost_dir.joinpath("lightgbm.joblib").write_bytes(b"not a model")
    save_registry({"features": {"use_deep": False},
                   "models": {"xgboost": {"threshold": .5}, "lightgbm": {"threshold": .5},
                              COMMITTEE_KEY: {"threshold": .5}}})
    suite = BoostingSuite.load()
    assert suite.keys == ["xgboost"]            # committee needs all three members
    json.loads(boost_dir.joinpath("registry.json").read_text())
