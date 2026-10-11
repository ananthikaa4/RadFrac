"""Cross-method explainability agreement: Grad-CAM vs occlusion sensitivity.

Produces, per sampled test image, a 3-panel figure (original / Grad-CAM /
occlusion) annotated with the agreement statistics, plus an aggregate JSON so
the report can state how often the two independent methods localise the same
region.

Usage::

    python -m scripts.explain_agreement --n 4 --model cnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.preprocessing import load_splits, splits_exist  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM vs occlusion agreement")
    ap.add_argument("--n", type=int, default=3, help="images per class")
    ap.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    ap.add_argument("--patch", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.explain_dual import compare_maps, make_predict_fn, occlusion_attribution
    from src.gradcam import compute_gradcam
    from src.inference import FracturePredictor
    from src.preprocessing import load_image_array, preprocess_for_dl

    if not splits_exist():
        raise SystemExit("No splits. Run 'python -m src.preprocessing' first.")
    _, _, test = load_splits()

    predictor = FracturePredictor(args.model)
    predict_fn = make_predict_fn(args.model, predictor)

    out_dir = config.FIGURES_DIR / "explain_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = []
    for label in range(len(config.CLASS_NAMES)):
        picked.extend([s for s in test if s.label == label][: args.n])

    records = []
    for i, sample in enumerate(picked):
        img_uint8 = load_image_array(sample.filepath)       # already model-size
        img01 = preprocess_for_dl(img_uint8)

        gcam = compute_gradcam(predictor._model, img01)
        occl = occlusion_attribution(predict_fn, img_uint8,
                                     patch=args.patch, stride=args.stride)
        agree = compare_maps(gcam, occl)
        prob = predict_fn(img_uint8)

        records.append({
            "file": Path(sample.filepath).name,
            "true_label": config.CLASS_NAMES[sample.label],
            "probability": round(float(prob), 4),
            **agree.to_dict(),
        })

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
        axes[0].imshow(img_uint8)
        axes[0].set_title(f"true={config.CLASS_NAMES[sample.label]}\nP={prob:.2f}",
                          fontsize=9)
        axes[1].imshow(img_uint8)
        axes[1].imshow(gcam, cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM (gradient-based)", fontsize=9)
        axes[2].imshow(img_uint8)
        axes[2].imshow(occl, cmap="jet", alpha=0.45)
        axes[2].set_title("Occlusion (black-box)", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"agreement: r={agree.correlation:.2f}  IoU={agree.iou:.2f}  "
            f"peak-dist={agree.peak_distance:.2f}  -> "
            f"{'AGREE' if agree.agree else 'DISAGREE'}",
            fontsize=9,
        )
        fig.tight_layout()
        path = out_dir / f"agreement_{args.model}_{i:02d}_{config.CLASS_NAMES[sample.label]}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  {path.name}: r={agree.correlation:.3f} IoU={agree.iou:.3f} "
              f"{'AGREE' if agree.agree else 'DISAGREE'}")

    # Only conclusive cases can support an agreement claim. Flat-map cases
    # (typically confident negatives, where occlusion has nothing to remove)
    # are reported separately rather than counted as agreement.
    conclusive = [r for r in records if r.get("conclusive", True)]
    n_agree = sum(1 for r in conclusive if r["agree"])
    summary = {
        "model": args.model,
        "n_images": len(records),
        "n_conclusive": len(conclusive),
        "n_inconclusive": len(records) - len(conclusive),
        "n_agree": n_agree,
        "agreement_rate_over_conclusive": (
            round(n_agree / len(conclusive), 3) if conclusive else None),
        "mean_correlation_conclusive": (
            round(float(np.mean([r["correlation"] for r in conclusive])), 4)
            if conclusive else None),
        "mean_iou_conclusive": (
            round(float(np.mean([r["iou"] for r in conclusive])), 4)
            if conclusive else None),
        "methods": ["grad-cam (gradient-based)", "occlusion sensitivity (black-box)"],
        "note": ("Inconclusive cases produced a flat attribution map (no signal "
                 "to compare) and are excluded from the agreement rate rather "
                 "than counted as agreeing."),
        "per_image": records,
    }
    out_json = config.METRICS_DIR / "explain_agreement.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"\nConclusive cases: {len(conclusive)}/{len(records)} "
          f"({summary['n_inconclusive']} had no attribution signal)")
    if conclusive:
        print(f"Agreement on conclusive cases: {n_agree}/{len(conclusive)} "
              f"({summary['agreement_rate_over_conclusive']:.0%}), "
              f"mean r={summary['mean_correlation_conclusive']}, "
              f"mean IoU={summary['mean_iou_conclusive']}")
    print(f"figures -> {out_dir}")
    print(f"summary -> {out_json}")


if __name__ == "__main__":
    main()
