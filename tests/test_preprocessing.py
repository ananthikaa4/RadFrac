"""Unit tests for data discovery, splitting and image preprocessing."""
from __future__ import annotations

import numpy as np

from src import config
from src.preprocessing import (
    compute_class_weights,
    discover_samples,
    load_dl_input,
    load_image_array,
    load_splits,
    preprocess_for_dl,
    save_splits,
    splits_exist,
    stratified_split,
)


def test_discover_samples_finds_both_classes(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    assert len(samples) == 16
    labels = {s.label for s in samples}
    assert labels == {0, 1}


def test_stratified_split_disjoint_and_proportioned(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    train, val, test = stratified_split(samples)
    # No overlap between partitions.
    paths_train = {s.filepath for s in train}
    paths_val = {s.filepath for s in val}
    paths_test = {s.filepath for s in test}
    assert paths_train.isdisjoint(paths_val)
    assert paths_train.isdisjoint(paths_test)
    assert paths_val.isdisjoint(paths_test)
    assert len(train) + len(val) + len(test) == len(samples)
    # Every split has both classes (stratification).
    for part in (train, val, test):
        assert {s.label for s in part} == {0, 1}


def test_save_and_load_splits_roundtrip(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    train, val, test = stratified_split(samples)
    save_splits(train, val, test)
    assert splits_exist()
    r_train, r_val, r_test = load_splits()
    assert [s.filepath for s in r_train] == [s.filepath for s in train]
    assert [s.label for s in r_test] == [s.label for s in test]


def test_load_image_array_shape_and_dtype(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    arr = load_image_array(samples[0].filepath)
    assert arr.shape == (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    assert arr.dtype == np.uint8


def test_preprocess_for_dl_range(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    arr = preprocess_for_dl(load_image_array(samples[0].filepath))
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_load_dl_input_is_batched(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    x = load_dl_input(samples[0].filepath)
    assert x.shape == (1, *config.IMAGE_SIZE, config.IMAGE_CHANNELS)


def test_class_weights_sum_reasonable(tiny_raw_dataset):
    samples = discover_samples(tiny_raw_dataset)
    train, _, _ = stratified_split(samples)
    w = compute_class_weights(train)
    assert set(w.keys()) == {0, 1}
    assert all(v > 0 for v in w.values())
