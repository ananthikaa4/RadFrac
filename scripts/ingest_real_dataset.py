"""Reorganise a downloaded fracture dataset into RadFrac's expected layout.

Real Kaggle/FracAtlas zips come in many shapes (train/val/test splits, nested
body-part folders, or a CSV of labels). This script scans an extracted dataset
directory, infers each image's class from its folder path (or a CSV), and copies
the images into::

    data/raw/fractured/
    data/raw/not_fractured/

It then removes the synthetic-data marker so downstream tools stop showing the
synthetic warning. It never deletes your source files.

Examples
--------
Folder-based (e.g. Kaggle "fracture-multi-region-x-ray-data" after --unzip)::

    python -m scripts.ingest_real_dataset --source data/raw/_download

CSV-based (e.g. FracAtlas)::

    python -m scripts.ingest_real_dataset \
        --csv data/raw/FracAtlas/dataset.csv \
        --image-dir data/raw/FracAtlas/images \
        --filename-col image_id --label-col fractured --positive-value 1

After ingesting::

    python -m src.preprocessing
    python -m src.train --model all
    python -m src.evaluate
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dataset_status import remove_marker  # noqa: E402

# Folder-name tokens that map to each class. Matched case-insensitively as a
# substring of any path component.
POSITIVE_TOKENS = ("fractured", "fracture", "positive", "abnormal", "yes", "broken")
NEGATIVE_TOKENS = ("not_fractured", "notfractured", "non_fractured", "nonfractured",
                   "normal", "negative", "healthy", "no", "unbroken")


def _classify_by_path(path: Path) -> int | None:
    """Infer class from folder names in the path. Returns label or None."""
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    # Check negatives first: "not_fractured" contains "fractured" as substring.
    if any(tok in joined for tok in NEGATIVE_TOKENS):
        neg_hit = True
    else:
        neg_hit = False
    pos_hit = any(f"/{tok}" in f"/{joined}" or tok in parts for tok in POSITIVE_TOKENS)
    # Prefer the most specific: an explicit negative token wins over a bare
    # "fracture" appearing in a dataset name.
    if neg_hit and not any(tok in parts for tok in POSITIVE_TOKENS):
        return config.CLASS_NAMES.index("not_fractured")
    if pos_hit:
        return config.POSITIVE_INDEX
    if neg_hit:
        return config.CLASS_NAMES.index("not_fractured")
    return None


def _copy(src: Path, label: int, counters: dict[int, int], move: bool) -> None:
    class_name = config.CLASS_NAMES[label]
    dest_dir = config.RAW_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Neutral prefix so real images never match the synthetic filename pattern.
    dest = dest_dir / (f"{config.REAL_IMAGE_PREFIX}"
                       f"{counters[label]:06d}{src.suffix.lower()}")
    while dest.exists():                      # safe to re-run
        counters[label] += 1
        dest = dest_dir / (f"{config.REAL_IMAGE_PREFIX}"
                           f"{counters[label]:06d}{src.suffix.lower()}")
    counters[label] += 1
    if move:
        shutil.move(str(src), dest)
    else:
        shutil.copy2(src, dest)


def ingest_folder(source: Path, move: bool) -> dict[int, int]:
    counters = {i: 0 for i in range(len(config.CLASS_NAMES))}
    skipped = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in config.VALID_EXTENSIONS:
            continue
        # Don't re-ingest images already in the target folders.
        if config.RAW_DIR in path.parents and path.parent.name in config.CLASS_NAMES:
            continue
        label = _classify_by_path(path)
        if label is None:
            skipped += 1
            continue
        _copy(path, label, counters, move)
    if skipped:
        print(f"  {skipped} images skipped (class could not be inferred from path)")
    return counters


def ingest_csv(csv_path: Path, image_dir: Path, filename_col: str,
               label_col: str, positive_value: str, move: bool) -> dict[int, int]:
    counters = {i: 0 for i in range(len(config.CLASS_NAMES))}
    missing = 0
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            fname = row[filename_col]
            src = image_dir / fname
            if not src.exists():
                missing += 1
                continue
            is_pos = str(row[label_col]).strip() == str(positive_value)
            label = config.POSITIVE_INDEX if is_pos else config.CLASS_NAMES.index("not_fractured")
            _copy(src, label, counters, move)
    if missing:
        print(f"  {missing} CSV rows skipped (image file not found)")
    return counters


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a real fracture dataset")
    ap.add_argument("--source", help="extracted dataset dir (folder-based layout)")
    ap.add_argument("--csv", help="label CSV (CSV-based layout, e.g. FracAtlas)")
    ap.add_argument("--image-dir", help="image dir for --csv mode")
    ap.add_argument("--filename-col", default="image_id")
    ap.add_argument("--label-col", default="fractured")
    ap.add_argument("--positive-value", default="1")
    ap.add_argument("--move", action="store_true",
                    help="move files instead of copying (saves disk space)")
    args = ap.parse_args()

    if not args.source and not args.csv:
        ap.error("provide --source (folder mode) or --csv + --image-dir (CSV mode)")

    if args.csv:
        if not args.image_dir:
            ap.error("--csv requires --image-dir")
        counters = ingest_csv(Path(args.csv), Path(args.image_dir),
                              args.filename_col, args.label_col,
                              args.positive_value, args.move)
    else:
        source = Path(args.source)
        if not source.is_dir():
            ap.error(f"source dir not found: {source}")
        counters = ingest_folder(source, args.move)

    total = sum(counters.values())
    if total == 0:
        raise SystemExit(
            "No images were ingested. Check the source layout — for folder mode "
            "each image's path must contain a class token (e.g. 'fractured' / "
            "'normal'); for CSV mode check the column names and --positive-value."
        )

    # Real data now present -> drop the synthetic marker so warnings clear.
    remove_marker()

    print("Ingested:")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name}: {counters[i]}")
    print(f"Total: {total} images -> {config.RAW_DIR}")
    print("Synthetic marker removed. Next: python -m src.preprocessing")


if __name__ == "__main__":
    main()
