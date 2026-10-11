"""Knowledge distillation (4.3) - shrink the model for low-cost clinic hardware.

Why this is not a generic add-on here
------------------------------------
RadFrac's stated deployment target is a rural PHC: offline, modest, probably
non-GPU hardware. That is a hard constraint, not a preference. Distillation
addresses it directly - train a larger accurate "teacher", then train a much
smaller "student" to reproduce the teacher's *soft* output distribution rather
than just the hard labels. The soft targets carry more information per example
(they encode how confident the teacher is and which cases are borderline), so
the student typically lands closer to teacher accuracy than the same small
network trained on labels alone.

Loss: ``alpha * KL(student_soft || teacher_soft) * T^2 + (1-alpha) * BCE(student, y)``

The ``T^2`` factor restores gradient magnitude after temperature softening, per
Hinton et al. (2015).
"""
from __future__ import annotations

from . import config


def build_student(input_shape=None, width: float = 0.5,
                  learning_rate: float = 1e-3):
    """A deliberately small CNN for constrained hardware.

    ``width`` scales the channel counts relative to the standard trunk, so
    ``width=0.5`` gives roughly a quarter of the convolutional parameters.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if input_shape is None:
        input_shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)

    filters = [max(4, int(f * width)) for f in (16, 32, 64)]
    model = models.Sequential(name="radfrac_student")
    model.add(layers.Input(shape=input_shape))
    for f in filters:
        model.add(layers.Conv2D(f, 3, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_distillation_loss(alpha: float = 0.7, temperature: float = 4.0):
    """Combined soft/hard distillation loss.

    ``y_true`` is packed as ``[hard_label, teacher_probability]`` along the last
    axis, which lets us use a plain ``model.compile(loss=...)`` instead of
    subclassing ``keras.Model``. Precomputing the teacher's outputs once (rather
    than re-running it every batch) is both faster and avoids Keras 3's
    sub-model attribute-tracking pitfalls.
    """
    import tensorflow as tf

    eps = 1e-7
    T = float(temperature)

    def soften(p):
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        return tf.sigmoid(tf.math.log(p / (1.0 - p)) / T)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        hard_y = y_true[:, 0:1]
        teacher_p = y_true[:, 1:2]
        s_prob = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # Hard-label cross entropy.
        hard = tf.keras.losses.binary_crossentropy(hard_y, s_prob)

        # Temperature-softened Bernoulli KL divergence (teacher || student).
        ts = tf.clip_by_value(soften(teacher_p), eps, 1.0 - eps)
        ss = tf.clip_by_value(soften(s_prob), eps, 1.0 - eps)
        kl = (ts * tf.math.log(ts / ss)
              + (1.0 - ts) * tf.math.log((1.0 - ts) / (1.0 - ss)))
        soft = tf.reduce_mean(kl, axis=-1) * (T ** 2)

        return alpha * soft + (1.0 - alpha) * tf.reshape(hard, tf.shape(soft))

    loss_fn.__name__ = "distillation_loss"
    return loss_fn


def compile_student_for_distillation(student, alpha: float = 0.7,
                                     temperature: float = 4.0,
                                     learning_rate: float = 1e-3):
    """Recompile a student model with the distillation loss.

    Metrics are computed against the hard label (column 0 of the packed target)
    so the reported accuracy/AUC stay interpretable.
    """
    import tensorflow as tf

    def hard_accuracy(y_true, y_pred):
        return tf.keras.metrics.binary_accuracy(
            tf.cast(y_true[:, 0:1], y_pred.dtype), y_pred)

    hard_accuracy.__name__ = "accuracy"

    auc_metric = tf.keras.metrics.AUC(name="auc")

    class HardAUC(tf.keras.metrics.AUC):
        def update_state(self, y_true, y_pred, sample_weight=None):
            return super().update_state(y_true[:, 0:1], y_pred, sample_weight)

    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=make_distillation_loss(alpha, temperature),
        metrics=[hard_accuracy, HardAUC(name="auc")],
    )
    return student


def make_distillation_dataset(samples, teacher, batch_size: int | None = None,
                              training: bool = False):
    """Dataset yielding ``(image, [hard_label, teacher_prob])``.

    The teacher is evaluated once up front, so training the student never pays
    the teacher's forward cost again.
    """
    import numpy as np
    import tensorflow as tf

    from .preprocessing import make_tf_dataset

    batch_size = batch_size or config.BATCH_SIZE

    # Teacher soft targets in a fixed (unshuffled) order.
    ordered = make_tf_dataset(samples, batch_size=batch_size, training=False,
                              shuffle=False)
    teacher_probs = teacher.predict(ordered, verbose=0).reshape(-1)
    hard = np.array([s.label for s in samples], dtype="float32")
    packed = np.stack([hard, teacher_probs.astype("float32")], axis=1)

    paths = [s.filepath for s in samples]

    def load(path, target):
        data = tf.io.read_file(path)
        img = tf.io.decode_image(data, channels=config.IMAGE_CHANNELS,
                                 expand_animations=False)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([*config.IMAGE_SIZE, config.IMAGE_CHANNELS])
        return img, target

    ds = tf.data.Dataset.from_tensor_slices((paths, packed))
    if training:
        ds = ds.shuffle(max(len(paths), 1), seed=config.RANDOM_SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        def aug(img, target):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            return tf.clip_by_value(img, 0.0, 1.0), target
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def model_footprint(model) -> dict:
    """Parameter count and approximate float32 size, for the hardware argument."""
    params = int(model.count_params())
    return {
        "parameters": params,
        "approx_float32_mb": round(params * 4 / (1024 ** 2), 3),
    }
