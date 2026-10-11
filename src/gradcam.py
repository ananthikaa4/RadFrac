"""Grad-CAM / Grad-CAM++ explainability for the deep RadFrac models.

Produces a class-activation heatmap over the image regions that drove the
"fractured" score.

Four things matter for localisation quality here, in order of measured impact on
this project's models:

1. **Differentiate the logit, not the sigmoid probability.** This is the big one.
   The CNN reaches p ≈ 1.0 on real fractures, so the sigmoid slope ``p(1-p)``
   collapses (measured as low as 3.6e-07) and the gradients flowing back become
   float noise (5.4e-10). The channel weights are then meaningless and the map
   looks scattered *precisely on the confident cases you most want to explain*.
   Recovering the logit as ``log(p/(1-p))`` exactly cancels the sigmoid
   derivative and restored gradient magnitudes by up to ~2.8e6x on real images.

2. **Tap the last conv feature map, not a pooling layer after it.** The custom
   CNN's deepest conv activation is 16x16, but the following MaxPooling2D is
   8x8. Targeting the pooling output throws away half the linear spatial
   precision before Grad-CAM ever runs - costly when a hairline fracture is a
   small region.

3. **Grad-CAM++ weighting.** Vanilla Grad-CAM averages the gradient over each
   channel, which favours large blobs. Grad-CAM++ uses a weighted combination of
   *positive* partial derivatives, which localises small discriminative regions
   more tightly.

4. **Bicubic upsampling** of the low-resolution map, which avoids the blocky
   edges of nearest/linear interpolation.

Both ``method="gradcam"`` and ``method="gradcam++"`` are available so old and new
behaviour can be compared directly.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from . import config

# Layer types that are NOT acceptable Grad-CAM targets even though their output
# is 4-D: pooling discards spatial precision, dropout/rescaling carry no features.
_NON_FEATURE_4D = (
    "MaxPooling2D", "AveragePooling2D", "MaxPool2D", "AvgPool2D",
    "Dropout", "SpatialDropout2D", "Rescaling", "Normalization",
    "ZeroPadding2D", "Cropping2D", "Resizing",
)


def find_gradcam_target_layer(model, prefer_resolution: bool = True) -> str:
    """Return the name of the layer Grad-CAM should tap.

    Walks the model backwards and returns the deepest layer with a 4-D output
    that actually carries convolutional features - skipping pooling/dropout
    wrappers, whose outputs are spatially degraded copies of the layer before
    them.

    For a model containing a nested backbone (the MobileNetV2 transfer model),
    the wrapping backbone layer is returned, since that is its final feature map.
    """
    import tensorflow as tf

    candidates: list[tuple[str, tuple]] = []
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            continue
        if len(shape) != 4:
            continue
        # A nested model (e.g. the MobileNetV2 backbone) is a valid target.
        if hasattr(layer, "layers"):
            return layer.name
        if type(layer).__name__ in _NON_FEATURE_4D:
            candidates.append((layer.name, shape))   # remember, but deprioritise
            continue
        return layer.name

    if candidates and not prefer_resolution:
        return candidates[0][0]
    if candidates:
        return candidates[0][0]
    raise ValueError("No convolutional feature map found for Grad-CAM.")


# Backwards-compatible alias (older callers/tests used the private name).
_find_last_conv_layer = find_gradcam_target_layer


def _build_grad_model(model, layer_name: str, input_shape, logit_output: bool = True):
    """Functional model returning (target activations, score).

    Built by threading a fresh Input through the layers, because a model
    deserialized from disk does not expose connected symbolic ``.output``
    tensors in Keras 3.

    When ``logit_output`` is set and the final layer is a sigmoid Dense, the
    second output is the TRUE pre-activation (``x @ W + b``) instead of the
    squashed probability. This matters more than it sounds: reconstructing the
    logit as ``log(p/(1-p))`` needs a clip to stay finite, and once the model
    saturates to p == 1.0 in float32 the clipped value is a CONSTANT, so its
    gradient is exactly zero and the CAM comes out all-zeros - on precisely the
    confident cases one most wants explained. Reading the pre-activation avoids
    the saturation entirely.
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=input_shape)
    y = inp
    tap = None
    layers = [l for l in model.layers
              if not isinstance(l, tf.keras.layers.InputLayer)]

    for i, layer in enumerate(layers):
        is_last = (i == len(layers) - 1)
        if (is_last and logit_output and isinstance(layer, tf.keras.layers.Dense)
                and layer.activation is not tf.keras.activations.linear):
            # Same Dense transform, linear activation: gives the raw logit.
            # A cloned layer is used rather than raw tf ops because Keras 3
            # forbids TensorFlow ops on symbolic KerasTensors.
            weights = layer.get_weights()
            linear = tf.keras.layers.Dense(
                layer.units, activation=None,
                use_bias=layer.use_bias,
                name=f"{layer.name}_logit",
            )
            y = linear(y)
            linear.set_weights(weights)
        else:
            y = layer(y)
        if layer.name == layer_name:
            tap = y
    if tap is None:
        raise ValueError(f"Layer {layer_name!r} not found while building grad model.")
    return tf.keras.models.Model(inp, [tap, y])


# --------------------------------------------------------------------------- #
# Grad-model cache
# --------------------------------------------------------------------------- #
# Building the grad model threads a fresh Input through every layer and
# constructs a new Dense plus a new Model. Doing that per request leaked a
# Keras graph on every /predict, which shows up in a container as steadily
# growing RSS and eventually an OOM kill rather than an error. The result
# depends only on (model, layer_name, input_shape, logit_output) - all constant
# at serving time - so it is built once and reused.
_GRAD_MODEL_CACHE: dict[tuple, Any] = {}
_GRAD_MODEL_LOCK = threading.Lock()


def _get_grad_model(model, layer_name: str, input_shape, logit_output: bool = True):
    key = (id(model), layer_name, tuple(input_shape), bool(logit_output))
    cached = _GRAD_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _GRAD_MODEL_LOCK:
        # Re-check inside the lock: Keras model construction is not
        # thread-safe, and two concurrent requests would otherwise each build
        # a full clone.
        cached = _GRAD_MODEL_CACHE.get(key)
        if cached is None:
            cached = _build_grad_model(model, layer_name, input_shape,
                                       logit_output=logit_output)
            _GRAD_MODEL_CACHE[key] = cached
    return cached


def clear_grad_model_cache() -> None:
    """Drop cached grad models (used by tests that rebuild models)."""
    with _GRAD_MODEL_LOCK:
        _GRAD_MODEL_CACHE.clear()


def _score_tensor(preds, already_logit: bool):
    """Scalar score to differentiate.

    When the grad model was built with ``logit_output=True`` the second output is
    already the pre-sigmoid activation, so it is used directly. Otherwise the
    raw sigmoid probability is differentiated, which is what the original
    implementation did (kept so old behaviour can still be reproduced for A/B
    comparison).
    """
    return preds[:, 0]


def compute_gradcam(
    model,
    img_float01: np.ndarray,
    layer_name: str | None = None,
    method: str = "layercam",
    use_logit: bool = True,
    interpolation: str = "cubic",
) -> np.ndarray:
    """Return a heatmap in [0, 1] at the input's spatial size.

    Parameters
    ----------
    img_float01 : single (H, W, C) image scaled to [0, 1].
    method      : ``"layercam"`` (default), ``"gradcam++"`` or ``"gradcam"``.
    use_logit   : differentiate the logit rather than the sigmoid probability.
    interpolation : ``"cubic"``, ``"linear"`` or ``"nearest"`` for upsampling.

    The default is LayerCAM rather than Grad-CAM++ because that is what the
    measurements on this project's models actually supported. At a matched
    16x16 tap and a fixed masking budget over real fractured test images,
    LayerCAM gave a larger deletion drop (0.635) than Grad-CAM++ (0.521) or
    vanilla Grad-CAM (0.550), while flagging less than half the area
    (0.015 vs 0.031) and producing the most peaked map. Pushing the tap to
    32x32 or 64x64 made maps prettier but far less faithful (deletion drop
    collapsed to 0.11-0.18), so 16x16 is the sweet spot.
    """
    import tensorflow as tf

    if layer_name is None:
        layer_name = find_gradcam_target_layer(model)

    x = tf.convert_to_tensor(img_float01[None, ...], dtype=tf.float32)
    grad_model = _get_grad_model(model, layer_name, tuple(img_float01.shape),
                                 logit_output=use_logit)

    method = method.lower().replace("_", "").replace("-", "")
    if method in ("gradcampp", "gradcam++", "gradcamplusplus"):
        heatmap = _gradcam_plusplus_map(grad_model, x, use_logit)
    elif method == "gradcam":
        heatmap = _gradcam_map(grad_model, x, use_logit)
    elif method == "layercam":
        heatmap = _layercam_map(grad_model, x, use_logit)
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'gradcam', 'gradcam++' or 'layercam'.")

    return _resize_map(heatmap, img_float01.shape[:2], interpolation)


# --------------------------------------------------------------------------- #
# Vanilla Grad-CAM
# --------------------------------------------------------------------------- #
def _gradcam_map(grad_model, x, use_logit: bool) -> np.ndarray:
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        tape.watch(conv_out)
        score = _score_tensor(preds, use_logit)
    grads = tape.gradient(score, conv_out)
    if grads is None:
        return np.zeros(conv_out.shape[1:3], dtype="float32")

    # One weight per channel: the spatially averaged gradient.
    weights = tf.reduce_mean(grads, axis=(1, 2))[0]          # (C,)
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)      # (H, W)
    return _normalise(tf.nn.relu(cam).numpy())


# --------------------------------------------------------------------------- #
# Grad-CAM++
# --------------------------------------------------------------------------- #
def _gradcam_plusplus_map(grad_model, x, use_logit: bool) -> np.ndarray:
    """Grad-CAM++ (Chattopadhyay et al., 2018).

    Channel weights come from a weighted combination of *positive* partial
    derivatives::

        alpha_ij^k = d2 / (2*d2 + sum_ab(A_ab^k) * d3)
        w^k        = sum_ij alpha_ij^k * relu(d1_ij^k)

    where d1/d2/d3 are the first/second/third derivatives of the score w.r.t.
    the activations. Weighting each spatial position individually (instead of a
    flat channel mean) is what sharpens localisation of small regions.
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        tape.watch(conv_out)
        score = _score_tensor(preds, use_logit)
    grads = tape.gradient(score, conv_out)
    if grads is None:
        return np.zeros(conv_out.shape[1:3], dtype="float32")

    A = conv_out[0]                 # (H, W, C)
    d1 = grads[0]                   # (H, W, C)

    # For a score of the form exp(s) the higher derivatives reduce to powers of
    # d1, which is the standard closed form used by Grad-CAM++ implementations.
    d2 = d1 * d1
    d3 = d2 * d1

    # sum over spatial positions per channel, broadcast back
    global_sum = tf.reduce_sum(A, axis=(0, 1), keepdims=True)    # (1, 1, C)
    denom = 2.0 * d2 + global_sum * d3
    denom = tf.where(tf.abs(denom) < 1e-12, tf.ones_like(denom), denom)
    alpha = d2 / denom

    weights = tf.reduce_sum(alpha * tf.nn.relu(d1), axis=(0, 1))  # (C,)
    cam = tf.reduce_sum(A * weights, axis=-1)                     # (H, W)
    return _normalise(tf.nn.relu(cam).numpy())


# --------------------------------------------------------------------------- #
# LayerCAM
# --------------------------------------------------------------------------- #
def _layercam_map(grad_model, x, use_logit: bool) -> np.ndarray:
    """LayerCAM (Jiang et al., 2021) - per-location, not per-channel, weights.

    Grad-CAM collapses each channel's gradient to a single number, which forces
    one weight to describe the whole spatial map. That is fine at 7x7 but wastes
    a higher-resolution tap. LayerCAM instead weights every *location*
    independently::

        cam_ij = sum_k relu(grad_ij^k) * A_ij^k

    Because the weighting is element-wise, it stays sharp when applied to
    shallower, higher-resolution feature maps - which is exactly what this
    project needs, since the model demonstrably relies on fine local detail that
    a 8x8 or 16x16 grid cannot express.
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        tape.watch(conv_out)
        score = _score_tensor(preds, use_logit)
    grads = tape.gradient(score, conv_out)
    if grads is None:
        return np.zeros(conv_out.shape[1:3], dtype="float32")

    weights = tf.nn.relu(grads[0])                  # (H, W, C) element-wise
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1)
    return _normalise(tf.nn.relu(cam).numpy())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalise(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    return cam.astype("float32")


_INTERP = {"cubic": "INTER_CUBIC", "linear": "INTER_LINEAR", "nearest": "INTER_NEAREST"}


def _resize_map(heatmap: np.ndarray, size_hw, interpolation: str = "cubic") -> np.ndarray:
    """Upsample the low-resolution CAM to the image size.

    Bicubic avoids the blocky staircase that nearest/linear produce when going
    from e.g. 16x16 to 128x128 (an 8x magnification).
    """
    import cv2

    h, w = size_hw
    flag = getattr(cv2, _INTERP.get(interpolation, "INTER_CUBIC"))
    out = cv2.resize(heatmap, (w, h), interpolation=flag)
    # Bicubic can overshoot slightly outside [0, 1].
    return np.clip(out, 0.0, 1.0)


def overlay_heatmap(img_float01: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, interpolation: str = "cubic") -> np.ndarray:
    """Blend a heatmap over the image, returning a uint8 RGB image."""
    import cv2

    img_uint8 = (np.clip(img_float01, 0, 1) * 255).astype("uint8")
    if heatmap.shape[:2] != img_uint8.shape[:2]:
        heatmap = _resize_map(heatmap, img_uint8.shape[:2], interpolation)
    heat_uint8 = (np.clip(heatmap, 0, 1) * 255).astype("uint8")
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1 - alpha, color, alpha, 0)
