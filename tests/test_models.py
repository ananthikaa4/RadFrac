"""Unit/integration tests for the three model families and inference."""
from __future__ import annotations

import numpy as np
import pytest

from src import config


def test_cnn_builds_and_predicts_probability():
    tf = pytest.importorskip("tensorflow")
    from src.model_cnn import build_cnn

    model = build_cnn()
    x = np.random.rand(2, *config.IMAGE_SIZE, config.IMAGE_CHANNELS).astype("float32")
    out = model.predict(x, verbose=0)
    assert out.shape == (2, 1)
    assert np.all((out >= 0) & (out <= 1))  # sigmoid probability


def test_classical_pipeline_fits_and_predicts(tiny_raw_dataset):
    pytest.importorskip("sklearn")
    from src.features_classical import build_feature_matrix
    from src.model_svm import build_classical, predict_proba_positive
    from src.preprocessing import discover_samples, stratified_split

    samples = discover_samples(tiny_raw_dataset)
    train, val, test = stratified_split(samples)
    X, y = build_feature_matrix(train + val)
    pipe = build_classical("svm")
    pipe.fit(X, y)

    Xt, yt = build_feature_matrix(test)
    probs = predict_proba_positive(pipe, Xt)
    assert probs.shape[0] == len(test)
    assert np.all((probs >= 0) & (probs <= 1))


def test_classical_save_load_roundtrip(tiny_raw_dataset, tmp_path):
    from src.features_classical import build_feature_matrix
    from src.model_svm import build_classical, load_classical, save_classical
    from src.preprocessing import discover_samples

    samples = discover_samples(tiny_raw_dataset)
    X, y = build_feature_matrix(samples)
    pipe = build_classical("svm")
    pipe.fit(X, y)

    path = tmp_path / "clf.pkl"
    save_classical(pipe, path)
    loaded = load_classical(path)
    assert np.allclose(pipe.predict_proba(X), loaded.predict_proba(X))


def test_metrics_computation():
    from src.evaluate import compute_metrics

    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0
    assert np.array(m["confusion_matrix"]).sum() == 5


def test_end_to_end_train_eval_infer_cnn(tiny_raw_dataset):
    """Full pipeline on synthetic-tiny data: split -> train CNN -> save ->
    load via inference -> predict. Guards against train/serve skew."""
    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(0)

    from src.inference import FracturePredictor
    from src.preprocessing import (
        discover_samples,
        make_tf_dataset,
        save_splits,
        stratified_split,
    )
    from src.model_cnn import build_cnn

    samples = discover_samples(tiny_raw_dataset)
    train, val, test = stratified_split(samples)
    save_splits(train, val, test)

    model = build_cnn()
    model.fit(make_tf_dataset(train, training=True), epochs=1, verbose=0)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.CNN_MODEL_FILE)

    predictor = FracturePredictor("cnn")
    pred = predictor.predict_file(test[0].filepath)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.label in config.CLASS_NAMES
    assert pred.model_key == "cnn"
