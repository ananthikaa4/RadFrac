"""Model disagreement detection (4.7).

RadFrac trains three independent models on the same data using genuinely
different inductive biases (a CNN learned from scratch, an ImageNet transfer
backbone, and hand-crafted HOG/Canny features + SVM). That diversity is not just
a report table - it is a free safety signal.

When all three agree, the screen is corroborated by three different ways of
looking at the image. When they disagree, that is precisely the case a
non-specialist should escalate rather than trust. We surface it explicitly
instead of silently picking the "best" model's answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from . import config


@dataclass
class DisagreementResult:
    flagged: bool
    spread: float                               # max prob - min prob
    labels: dict = field(default_factory=dict)  # model_key -> label
    probabilities: dict = field(default_factory=dict)
    n_models: int = 0
    majority_label: str | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def assess(per_model: dict[str, dict]) -> DisagreementResult:
    """Assess agreement across models.

    ``per_model`` maps model_key -> {"probability": float, "label": str}.
    Flags when the models split on the binary verdict, or when they nominally
    agree but their probabilities span more than ``DISAGREEMENT_SPREAD``
    (weak/fragile consensus worth a human look).
    """
    if not per_model:
        return DisagreementResult(flagged=False, spread=0.0, n_models=0,
                                  reason="no models available")

    probs = {k: float(v["probability"]) for k, v in per_model.items()}
    labels = {k: v["label"] for k, v in per_model.items()}
    spread = max(probs.values()) - min(probs.values())
    distinct = set(labels.values())

    # Majority vote (ties -> None).
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[lab] = counts.get(lab, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    majority = ordered[0][0] if (len(ordered) == 1 or ordered[0][1] > ordered[1][1]) else None

    if len(per_model) < 2:
        return DisagreementResult(
            flagged=False, spread=0.0, labels=labels, probabilities=probs,
            n_models=len(per_model), majority_label=majority,
            reason="only one model available - no cross-check possible",
        )

    if len(distinct) > 1:
        split = ", ".join(f"{k}={labels[k]}" for k in sorted(labels))
        return DisagreementResult(
            flagged=True, spread=spread, labels=labels, probabilities=probs,
            n_models=len(per_model), majority_label=majority,
            reason=f"models disagree on the verdict ({split}) - recommend manual review",
        )

    if spread > config.DISAGREEMENT_SPREAD:
        return DisagreementResult(
            flagged=True, spread=spread, labels=labels, probabilities=probs,
            n_models=len(per_model), majority_label=majority,
            reason=(f"models agree on '{next(iter(distinct))}' but confidence spans "
                    f"{spread:.2f} - weak consensus, recommend manual review"),
        )

    return DisagreementResult(
        flagged=False, spread=spread, labels=labels, probabilities=probs,
        n_models=len(per_model), majority_label=majority,
        reason=f"all {len(per_model)} models agree ({next(iter(distinct))})",
    )
