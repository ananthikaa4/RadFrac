"""Generate Grad-CAM overlay images for a few test samples.

Picks ``--n`` samples per class from the held-out test split, runs the trained
CNN (preferred) or transfer model, and saves side-by-side original / heatmap
overlays into ``outputs/figures/gradcam/``.

Usage::

    python -m scripts.generate_gradcam_samples --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.preprocessing import load_splits, load_image_array, preprocess_for_dl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM samples")
    parser.add_argument("--n", type=int, default=3, help="samples per class")
    parser.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    parser.add_argument("--compare", action="store_true",
                        help="render OLD vs NEW side by side for A/B review")
    parser.add_argument("--method", default="layercam",
                        choices=["layercam", "gradcam++", "gradcam"])
    parser.add_argument("--subdir", default=None,
                        help="write into outputs/figures/gradcam/<subdir>/")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from src.gradcam import compute_gradcam, overlay_heatmap

    model_file = config.CNN_MODEL_FILE if args.model == "cnn" else config.TRANSFER_MODEL_FILE
    if not model_file.exists():
        raise SystemExit(f"Model artifact {model_file} not found. Train first.")

    _, _, test = load_splits()
    if not test:
        raise SystemExit("Empty test split; run preprocessing/train first.")

    model = tf.keras.models.load_model(model_file)

    out_dir = config.FIGURES_DIR / "gradcam"
    if args.subdir:
        out_dir = out_dir / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Randomly sample per class. A head-slice is unrepresentative because
    # filenames are ordered by ingestion (and therefore by body region).
    rng = np.random.default_rng(config.RANDOM_SEED)
    picked = []
    for label in range(len(config.CLASS_NAMES)):
        subset = [s for s in test if s.label == label]
        if not subset:
            continue
        idx = rng.choice(len(subset), min(args.n, len(subset)), replace=False)
        picked.extend(subset[i] for i in idx)

    # The pre-fix configuration, kept so the change can be reviewed visually.
    OLD_KW = dict(method="gradcam", use_logit=False, interpolation="linear")
    old_layer = "max_pooling2d_3" if args.model == "cnn" else None

    for i, sample in enumerate(picked):
        img_uint8 = load_image_array(sample.filepath)
        img01 = preprocess_for_dl(img_uint8)
        prob = float(model.predict(img01[None, ...], verbose=0).reshape(-1)[0])

        new_heat = compute_gradcam(model, img01, method=args.method,
                                   use_logit=True, interpolation="cubic")
        new_overlay = overlay_heatmap(img01, new_heat)

        if args.compare:
            try:
                old_heat = compute_gradcam(model, img01, layer_name=old_layer, **OLD_KW)
                old_overlay = overlay_heatmap(img01, old_heat, interpolation="linear")
            except Exception as exc:
                print(f"    (old variant unavailable: {exc})")
                old_overlay = img_uint8

            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
            axes[0].imshow(img_uint8)
            axes[0].set_title(f"original\ntrue={config.CLASS_NAMES[sample.label]}",
                              fontsize=9)
            axes[1].imshow(old_overlay)
            axes[1].set_title("OLD: Grad-CAM @8x8 pool\nsigmoid, linear", fontsize=9)
            axes[2].imshow(new_overlay)
            axes[2].set_title(f"NEW: {args.method} @16x16 conv\nlogit, cubic",
                              fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"P(fractured)={prob:.3f}", fontsize=10)
            name = (f"compare_{args.model}_{i:02d}_"
                    f"{config.CLASS_NAMES[sample.label]}.png")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
            axes[0].imshow(img_uint8)
            axes[0].set_title(f"true={config.CLASS_NAMES[sample.label]}")
            axes[1].imshow(new_overlay)
            axes[1].set_title(f"P(fractured)={prob:.2f}")
            for ax in axes:
                ax.axis("off")
            name = (f"gradcam_{args.model}_{i:02d}_"
                    f"{config.CLASS_NAMES[sample.label]}.png")

        fig.tight_layout()
        out_path = out_dir / name
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  wrote {out_path.name}")

    print(f"Grad-CAM overlays -> {out_dir}")


if __name__ == "__main__":
    main()
