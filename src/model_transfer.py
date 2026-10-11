"""Model 2 - transfer learning with MobileNetV2.

MobileNetV2 pretrained on ImageNet is used as a frozen feature extractor with a
small classification head on top. Weights download automatically on first use;
if there is no network access, ``build_transfer`` falls back to randomly
initialised weights so the pipeline still runs (this is logged clearly).
"""
from __future__ import annotations

from . import config


def build_transfer(input_shape=None, learning_rate: float = config.LR_TRANSFER):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    if input_shape is None:
        input_shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)

    try:
        base = MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        weights_source = "imagenet"
    except Exception as exc:  # no network / download failure
        print(f"[transfer] Could not load ImageNet weights ({exc}); "
              f"falling back to random initialisation.")
        base = MobileNetV2(
            input_shape=input_shape, include_top=False, weights=None
        )
        weights_source = "random"

    base.trainable = False  # freeze the backbone for feature extraction

    inputs = layers.Input(shape=input_shape)
    # MobileNetV2 expects inputs scaled to [-1, 1]; our pipeline yields [0, 1].
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="radfrac_transfer")
    model.weights_source = weights_source  # type: ignore[attr-defined]

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
