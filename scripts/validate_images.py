"""Quarantine unreadable / corrupt images before training.

Real clinical datasets routinely contain a handful of truncated or malformed
files. TensorFlow's decoder raises inside the ``tf.data`` pipeline, which kills a
training run *mid-epoch* with an opaque
``jpeg::Uncompress failed. Invalid JPEG data`` error - after you have already
waited for the epoch. Screening them out up front is far cheaper.

Two checks per file, deliberately using the **same decoders the pipeline uses**
so this catches exactly what would break training and nothing else:

1. ``PIL.Image.open(...).verify()`` then a full ``load()`` - catches truncation.
2. ``tf.io.decode_image`` - catches anything TensorFlow specifically rejects
   (the authoritative check, since that is what ``make_tf_dataset`` calls).

Bad files are **moved**, not deleted, into ``data/raw/_bad_images/<class>/`` so
you can inspect them. Re-running is safe.

Usage::

    python -m scripts.validate_images            # dry run: report only
    python -m scripts.validate_images --apply    # quarantine bad files
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def check_file(path: Path) -> str | None:
    """Return None if the image is usable, else a short reason string."""
    # --- 1. PIL structural check -------------------------------------------
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()                 # cheap header/structure check
        with Image.open(path) as im:
            im.convert("RGB").load()    # forces full decode, catches truncation
    except Exception as exc:
        return f"PIL: {type(exc).__name__}: {str(exc)[:80]}"

    # --- 2. TensorFlow decode (authoritative for the training pipeline) ----
    try:
        import tensorflow as tf

        data = tf.io.read_file(str(path))
        img = tf.io.decode_image(data, channels=config.IMAGE_CHANNELS,
                                 expand_animations=False)
        if int(tf.size(img)) == 0:
            return "TF: decoded to an empty tensor"
    except Exception as exc:
        return f"TF: {type(exc).__name__}: {str(exc)[:80]}"

    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Quarantine corrupt images")
    ap.add_argument("--apply", action="store_true",
                    help="move bad files into data/raw/_bad_images/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    bad_root = config.RAW_DIR / "_bad_images"
    total = 0
    bad: list[tuple[Path, str]] = []

    for class_name in config.CLASS_NAMES:
        class_dir = config.RAW_DIR / class_name
        if not class_dir.is_dir():
            continue
        files = [p for p in sorted(class_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS]
        print(f"[{class_name}] checking {len(files)} files ...", flush=True)
        for i, p in enumerate(files, 1):
            total += 1
            reason = check_file(p)
            if reason:
                bad.append((p, reason))
                if not args.quiet:
                    print(f"    BAD  {p.name}: {reason}", flush=True)
            if i % 2000 == 0:
                print(f"    ... {i}/{len(files)}", flush=True)

    print(f"\nChecked {total} images; {len(bad)} unusable.")

    if not bad:
        print("All images decode cleanly - safe to train.")
        return

    if not args.apply:
        print("\nDRY RUN - re-run with --apply to quarantine these files.")
        return

    moved = 0
    for p, _reason in bad:
        dest_dir = bad_root / p.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        n = 1
        while dest.exists():
            dest = dest_dir / f"{p.stem}_{n}{p.suffix}"
            n += 1
        try:
            shutil.move(str(p), dest)
            moved += 1
        except OSError as exc:
            print(f"    could not move {p}: {exc}")

    print(f"Quarantined {moved} file(s) -> {bad_root}")
    print("\nRemaining usable images per class:")
    grand = 0
    for class_name in config.CLASS_NAMES:
        d = config.RAW_DIR / class_name
        n = len([p for p in d.iterdir()
                 if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS]) \
            if d.is_dir() else 0
        grand += n
        print(f"  {class_name:<16} {n}")
    print(f"  {'TOTAL':<16} {grand}")
    print("\nRe-run 'python -m src.preprocessing' to rebuild the splits.")


if __name__ == "__main__":
    main()
