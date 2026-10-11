"""Monte Carlo Dropout uncertainty estimation (4.1).

Why this matters for RadFrac's primary user: a general physician at a PHC with
no radiologist on site needs the tool to say "I am not sure about this one,
refer it" - and that is a *different* signal from a merely mid-range
probability. A single sigmoid output of 0.5 is ambiguous: it could mean the
model is confidently predicting a borderline case, or that it has no idea. Only
by sampling the model's predictive distribution can we distinguish those.

Implementation detail that matters for correctness
--------------------------------------------------
The naive approach is ``model(x, training=True)``, but that also puts
**BatchNormalization** into training mode, which recomputes batch statistics on
a single image and corrupts the prediction. Instead we clone the model, swapping
every ``Dropout`` layer for an always-on subclass, leaving BatchNorm in
inference mode. So the only stochasticity is dropout, which is exactly what MC
Dropout assumes.

Models without any dropout layer cannot yield MC uncertainty; that is reported
honestly via ``supported=False`` rather than fabricating a zero variance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config


@dataclass
class UncertaintyResult:
    mean: float               # mean P(fractured) across stochastic passes
    std: float                # std-dev across passes = epistemic uncertainty
    lower: float              # 5th percentile
    upper: float              # 95th percentile
    passes: int
    flagged: bool             # std exceeded the configured threshold
    supported: bool           # False when the model has no dropout layers
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _always_on_dropout_class():
    """Return a Dropout subclass that stays active during inference."""
    import tensorflow as tf

    class MCDropout(tf.keras.layers.Dropout):
        def call(self, inputs, training=None):  # noqa: D102
            return super().call(inputs, training=True)

    return MCDropout


def model_has_dropout(model) -> bool:
    import tensorflow as tf

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            return True
        # Look one level into nested models (e.g. a transfer backbone).
        if hasattr(layer, "layers"):
            for inner in layer.layers:
                if isinstance(inner, tf.keras.layers.Dropout):
                    return True
    return False


def build_mc_model(model):
    """Clone ``model`` with dropout forced on at inference.

    BatchNorm and every other layer keep inference behaviour; weights are copied
    across so the clone is numerically identical apart from active dropout.
    """
    import tensorflow as tf

    MCDropout = _always_on_dropout_class()

    def clone_fn(layer):
        if isinstance(layer, tf.keras.layers.Dropout):
            return MCDropout(rate=layer.rate, noise_shape=layer.noise_shape,
                             seed=layer.seed, name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    mc = tf.keras.models.clone_model(model, clone_function=clone_fn)
    mc.set_weights(model.get_weights())
    return mc


def mc_dropout_predict(
    model,
    img_float01: np.ndarray,
    passes: int | None = None,
    std_threshold: float | None = None,
    mc_model=None,
) -> UncertaintyResult:
    """Run ``passes`` stochastic forward passes and summarise the distribution.

    ``img_float01`` is a single (H, W, C) image scaled to [0, 1]. Pass a
    pre-built ``mc_model`` to avoid re-cloning on every request.
    """
    passes = passes or config.MC_DROPOUT_PASSES
    std_threshold = (config.UNCERTAINTY_STD_THRESHOLD
                     if std_threshold is None else std_threshold)

    if not model_has_dropout(model):
        return UncertaintyResult(
            mean=float("nan"), std=float("nan"), lower=float("nan"),
            upper=float("nan"), passes=0, flagged=False, supported=False,
            reason="model contains no Dropout layers; MC Dropout not applicable",
        )

    if mc_model is None:
        mc_model = build_mc_model(model)

    # Batch the same image `passes` times: one predict call, N stochastic draws.
    batch = np.repeat(img_float01[None, ...], passes, axis=0)
    preds = mc_model.predict(batch, verbose=0).reshape(-1).astype("float64")

    return UncertaintyResult(
        mean=float(preds.mean()),
        std=float(preds.std(ddof=0)),
        lower=float(np.percentile(preds, 5)),
        upper=float(np.percentile(preds, 95)),
        passes=int(passes),
        flagged=bool(preds.std(ddof=0) >= std_threshold),
        supported=True,
    )


def interpret(result: UncertaintyResult) -> str:
    """Short human-readable confidence statement for the screening note."""
    if not result.supported:
        return "uncertainty not available for this model"
    if result.flagged:
        return (f"high model uncertainty (sd={result.std:.3f} across "
                f"{result.passes} stochastic passes)")
    if result.std < 0.05:
        return f"stable prediction (sd={result.std:.3f})"
    return f"moderate uncertainty (sd={result.std:.3f})"
