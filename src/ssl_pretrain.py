"""Self-supervised contrastive pretraining (4.2) - SimCLR-style.

Why this belongs in RadFrac
---------------------------
The honest, structural weakness of every open fracture dataset is that the
*labelled* set is small (hundreds to a few thousand images), because labelling
requires radiologist time. Unlabelled X-rays, by contrast, are comparatively
plentiful. Contrastive self-supervision exploits exactly that asymmetry: it
learns a visual representation from unlabelled images, and only then fine-tunes
a small classifier head on the scarce labels. This is the mechanism behind
modern foundation models, applied here at a scale that fits the project.

Method (SimCLR)
---------------
1. Each image produces two independently augmented "views".
2. An encoder + projection head maps both into an embedding space.
3. The NT-Xent contrastive loss pulls the two views of the same image together
   while pushing apart views of different images in the batch.
4. The trained encoder is kept; the projection head is discarded.
5. A classification head is fine-tuned on the labelled training split.

No labels are used in step 1-4, so unlabelled data (including the test images'
*pixels*, if one were careful) could be used - but to keep the evaluation
unimpeachable this implementation pretrains on the **training split only**, so
there is provably no test-set leakage of any kind.
"""
from __future__ import annotations

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Augmentation: the core of contrastive learning
# --------------------------------------------------------------------------- #
def _augment_view(img):
    """Strong stochastic augmentation producing one contrastive view."""
    import tensorflow as tf

    x = img
    x = tf.image.random_flip_left_right(x)
    # Random resized crop: scale 0.6-1.0 of the original area.
    h, w = config.IMAGE_SIZE
    scale = tf.random.uniform([], 0.6, 1.0)
    ch = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    x = tf.image.random_crop(x, size=[ch, cw, config.IMAGE_CHANNELS])
    x = tf.image.resize(x, (h, w))
    x = tf.image.random_brightness(x, 0.25)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    # Radiographs are greyscale; keep colour jitter out and use blur instead.
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x


def make_contrastive_dataset(samples, batch_size: int = 32):
    """Dataset yielding (view1, view2) pairs. Labels are never used."""
    import tensorflow as tf

    paths = [s.filepath for s in samples]

    def load(path):
        data = tf.io.read_file(path)
        img = tf.io.decode_image(data, channels=config.IMAGE_CHANNELS,
                                 expand_animations=False)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([*config.IMAGE_SIZE, config.IMAGE_CHANNELS])
        return img

    def two_views(path):
        img = load(path)
        return _augment_view(img), _augment_view(img)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(max(len(paths), 1), seed=config.RANDOM_SEED)
    ds = ds.map(two_views, num_parallel_calls=tf.data.AUTOTUNE)
    # drop_remainder: NT-Xent needs a fixed batch size for its negatives.
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# Encoder + projection head
# --------------------------------------------------------------------------- #
def build_encoder(input_shape=None, name: str = "ssl_encoder"):
    """Convolutional encoder producing a flat embedding.

    Mirrors the custom CNN's trunk so the learned weights are transferable into
    the same downstream architecture family.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if input_shape is None:
        input_shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)

    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in (16, 32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D(name="embedding")(x)
    return models.Model(inp, x, name=name)


def build_projection_head(embed_dim: int, out_dim: int = 64):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(embed_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(out_dim)(x)
    return models.Model(inp, x, name="projection_head")


# --------------------------------------------------------------------------- #
# NT-Xent loss + SimCLR training model
# --------------------------------------------------------------------------- #
def nt_xent_loss(z1, z2, temperature: float = 0.1):
    """Normalised temperature-scaled cross entropy over a batch of 2N views."""
    import tensorflow as tf

    batch = tf.shape(z1)[0]
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    z = tf.concat([z1, z2], axis=0)                      # (2N, d)

    sim = tf.matmul(z, z, transpose_b=True) / temperature  # (2N, 2N)
    # Mask self-similarity on the diagonal.
    mask = tf.eye(2 * batch, dtype=sim.dtype) * -1e9
    sim = sim + mask

    # Positive pair for index i is i+N (and vice versa).
    idx = tf.range(batch)
    targets = tf.concat([idx + batch, idx], axis=0)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=sim)
    )


def build_simclr(encoder, projection_head, temperature: float = 0.1):
    """Wrap encoder + head in a Keras Model with a custom contrastive step."""
    import tensorflow as tf

    class SimCLR(tf.keras.Model):
        def __init__(self, encoder, head, temperature):
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.temperature = temperature
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.loss_tracker]

        def call(self, inputs, training=False):
            return self.head(self.encoder(inputs, training=training),
                             training=training)

        def train_step(self, data):
            v1, v2 = data
            with tf.GradientTape() as tape:
                z1 = self.head(self.encoder(v1, training=True), training=True)
                z2 = self.head(self.encoder(v2, training=True), training=True)
                loss = nt_xent_loss(z1, z2, self.temperature)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            v1, v2 = data
            z1 = self.head(self.encoder(v1, training=False), training=False)
            z2 = self.head(self.encoder(v2, training=False), training=False)
            loss = nt_xent_loss(z1, z2, self.temperature)
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

    return SimCLR(encoder, projection_head, temperature)


# --------------------------------------------------------------------------- #
# Fine-tuning the pretrained encoder for classification
# --------------------------------------------------------------------------- #
def build_finetune_model(encoder, learning_rate: float = 1e-3,
                        freeze_encoder: bool = False):
    """Attach a classification head to a (pre)trained encoder."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    encoder.trainable = not freeze_encoder
    inp = layers.Input(shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS))
    x = encoder(inp, training=not freeze_encoder)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="ssl_finetuned")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
