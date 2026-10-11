"""Model 1 - a small custom CNN trained from scratch.

Deliberately compact so it trains quickly on the small datasets typical of
fracture-detection coursework, while still being a genuine learned feature
extractor (four conv blocks + global average pooling).
"""
from __future__ import annotations

from . import config


def build_cnn(input_shape=None, learning_rate: float = config.LR_CNN):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if input_shape is None:
        input_shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)

    # BatchNorm is kept because it makes this from-scratch net optimise
    # reliably. Its known downside on small datasets is that the moving-average
    # statistics used at inference can shift the raw sigmoid outputs, so the
    # naive 0.5 cut-off is unreliable even when the model ranks well (high
    # AUC). We therefore DO NOT rely on 0.5: the pipeline selects each model's
    # decision threshold on the validation split (see src.train /
    # src.evaluate), which is standard practice and keeps the test set clean.
    model = models.Sequential(name="radfrac_cnn")
    model.add(layers.Input(shape=input_shape))

    for filters in (16, 32, 64, 128):
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))   # binary probability

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
