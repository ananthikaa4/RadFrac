"""Generate synthetic X-ray-like images to smoke-test the full pipeline.

This produces simple grayscale "bone" silhouettes. The ``fractured`` class has
a bright break line drawn across the bone; the ``not_fractured`` class does
not. Random noise, rotation and intensity jitter are added so the task is
non-trivial but clearly learnable.

**These are NOT real radiographs.** They exist only to verify that the loading,
training, evaluation and serving code runs end-to-end. Replace ``data/raw/``
with a genuine dataset before reporting any scientific results.

Usage::

    python -m scripts.make_synthetic_data --n_per_class 120
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
from PIL import Image, ImageDraw

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def _draw_bone(rng: np.random.Generator, size: int, fractured: bool) -> Image.Image:
    img = Image.new("L", (size, size), color=int(rng.integers(10, 30)))
    draw = ImageDraw.Draw(img)

    # Bone shaft: a vertical rounded bar, slightly rotated.
    cx = size // 2 + int(rng.integers(-8, 8))
    shaft_w = int(size * (0.16 + rng.random() * 0.06))
    top = int(size * (0.12 + rng.random() * 0.05))
    bottom = int(size * (0.88 - rng.random() * 0.05))
    intensity = int(rng.integers(170, 235))
    draw.rounded_rectangle(
        [cx - shaft_w, top, cx + shaft_w, bottom],
        radius=shaft_w, fill=intensity,
    )
    # Epiphyses (bulbs at the ends).
    bulb = int(shaft_w * 1.7)
    draw.ellipse([cx - bulb, top - bulb // 2, cx + bulb, top + bulb // 2], fill=intensity)
    draw.ellipse([cx - bulb, bottom - bulb // 2, cx + bulb, bottom + bulb // 2], fill=intensity)

    if fractured:
        # A dark break line across the shaft plus a small displacement.
        y = int(rng.integers(int(size * 0.35), int(size * 0.65)))
        jag = int(rng.integers(2, 6))
        break_color = int(rng.integers(5, 40))
        draw.line(
            [(cx - shaft_w - 2, y - jag), (cx + shaft_w + 2, y + jag)],
            fill=break_color, width=int(rng.integers(2, 4)),
        )

    arr = np.asarray(img).astype("float32")
    # Random rotation via PIL.
    angle = float(rng.uniform(-12, 12))
    img = Image.fromarray(arr.astype("uint8")).rotate(angle, fillcolor=int(arr.min()))
    arr = np.asarray(img).astype("float32")
    # Additive Gaussian noise.
    arr += rng.normal(0, 8, arr.shape)
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic X-ray-like data")
    parser.add_argument("--n_per_class", type=int, default=120)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--clean", action="store_true",
                        help="remove existing raw data first")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    for label, class_name in enumerate(config.CLASS_NAMES):
        out_dir = config.RAW_DIR / class_name
        if args.clean and out_dir.exists():
            # Only clear the class subfolder contents, not the (possibly locked)
            # parent raw/ directory itself.
            for f in out_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        fractured = (class_name == config.POSITIVE_CLASS)
        for i in range(args.n_per_class):
            img = _draw_bone(rng, args.size, fractured)
            # The "synthetic_" prefix makes these files unmistakable on disk, so
            # they can be detected and purged without any chance of catching a
            # real image.
            img.save(out_dir
                     / f"{config.SYNTHETIC_FILENAME_PREFIX}{class_name}_{i:04d}.png")
        print(f"  wrote {args.n_per_class} images -> {out_dir}")

    # Drop a hard marker so evaluate.py / the app / the report can detect this
    # data is synthetic and refuse to present its metrics as clinically valid.
    from src.dataset_status import write_marker  # noqa: E402
    write_marker(config.RAW_DIR, meta={
        "generator": "scripts/make_synthetic_data.py",
        "n_per_class": args.n_per_class,
        "seed": args.seed,
    })

    print(f"Synthetic dataset ready under {config.RAW_DIR} "
          f"({args.n_per_class} per class). "
          f"NOTE: synthetic data is for pipeline testing only.")


if __name__ == "__main__":
    main()
