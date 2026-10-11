"""Download and normalise the real fracture X-ray dataset from Kaggle.

Default target is the recommended primary dataset:
``bmadushanirodrigo/fracture-multi-region-x-ray-data`` (~9.2k train / 828 val /
506 test, pre-split into fractured / not-fractured folders across several
anatomical regions).

This script does three things:

1. Downloads + unzips via the Kaggle API (needs ``~/.kaggle/kaggle.json``).
2. Walks the extracted tree and normalises it into RadFrac's layout
   (``data/raw/fractured`` / ``data/raw/not_fractured``), tolerating the
   ``train/val/test`` sub-splits and the varied class folder spellings these
   dataset versions use ("fractured", "not fractured", "Not_Fractured", ...).
3. Removes the synthetic-data marker so the warning banners clear.

By default the dataset's own train/val/test folders are FLATTENED and RadFrac
re-splits them itself with a single stratified split, so every model shares
identical partitions. Pass ``--keep-original-splits`` to instead preserve the
publisher's split (written to ``data/splits`` directly).

Usage::

    python -m scripts.fetch_kaggle_dataset                 # download + normalise
    python -m scripts.fetch_kaggle_dataset --skip-download # normalise an existing download
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dataset_status import remove_marker  # noqa: E402

DEFAULT_SLUG = "bmadushanirodrigo/fracture-multi-region-x-ray-data"

# Folder-name tokens -> class. Checked most-specific-first so "not fractured"
# is never swallowed by the "fractured" substring.
NEGATIVE_TOKENS = (
    "not_fractured", "not fractured", "notfractured", "non_fractured",
    "non fractured", "nonfractured", "normal", "negative", "healthy",
    "no_fracture", "no fracture",
)
POSITIVE_TOKENS = ("fractured", "fracture", "positive", "abnormal", "broken")

SPLIT_HINTS = ("train", "val", "valid", "validation", "test")


def _kaggle_credentials_present() -> bool:
    for p in (Path.home() / ".kaggle" / "kaggle.json",
              Path.home() / ".config" / "kaggle" / "kaggle.json"):
        if p.exists():
            return True
    return False


def download(slug: str, dest: Path) -> None:
    if not _kaggle_credentials_present():
        raise SystemExit(
            "Kaggle API token not found.\n\n"
            "  1. Sign in at kaggle.com -> Settings -> API -> 'Create New API Token'\n"
            f"  2. Save the downloaded kaggle.json to: {Path.home() / '.kaggle' / 'kaggle.json'}\n"
            "  3. Re-run this command.\n\n"
            "Alternatively download the zip manually, extract it anywhere, and run:\n"
            "  python -m scripts.fetch_kaggle_dataset --skip-download --source <extracted-dir>"
        )
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", slug, "-p", str(dest), "--unzip"]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"Kaggle download failed (exit {result.returncode}). "
            "Check the dataset slug, your token, and network access."
        )


def classify(path: Path) -> int | None:
    """Infer class from the folder path, negatives first."""
    parts = [p.lower().replace("-", "_") for p in path.parts]
    joined = " / ".join(parts)
    for tok in NEGATIVE_TOKENS:
        if tok in joined:
            return config.CLASS_NAMES.index("not_fractured")
    for tok in POSITIVE_TOKENS:
        if tok in joined:
            return config.POSITIVE_INDEX
    return None


def normalise(source: Path, move: bool) -> dict[int, int]:
    counters = {i: 0 for i in range(len(config.CLASS_NAMES))}
    skipped: list[Path] = []

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in config.VALID_EXTENSIONS:
            continue
        # Never re-ingest files already sitting in the normalised destination
        # folders. This must compare the parent directory EXACTLY against
        # data/raw/<class>: a looser check on the folder *name* would also skip
        # the downloaded dataset's own "fractured"/"not fractured" folders and
        # silently ingest nothing.
        if path.parent in {config.RAW_DIR / c for c in config.CLASS_NAMES}:
            continue

        label = classify(path.relative_to(source))
        if label is None:
            skipped.append(path)
            continue

        class_name = config.CLASS_NAMES[label]
        out_dir = config.RAW_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Neutral "xr_" prefix: must not look like the synthetic generator's
        # output, or real data would trigger the synthetic-data warning.
        stem = f"{config.REAL_IMAGE_PREFIX}{counters[label]:06d}"
        dest = out_dir / f"{stem}{path.suffix.lower()}"
        # Idempotency: never clobber an existing file; skip if identical size.
        if dest.exists():
            while dest.exists():
                counters[label] += 1
                dest = out_dir / (f"{config.REAL_IMAGE_PREFIX}"
                                  f"{counters[label]:06d}{path.suffix.lower()}")
        counters[label] += 1
        if move:
            shutil.move(str(path), dest)
        else:
            shutil.copy2(path, dest)

    if skipped:
        print(f"  {len(skipped)} images skipped (class not inferable). Examples:")
        for p in skipped[:5]:
            print(f"    {p}")
    return counters


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + normalise a real fracture dataset")
    ap.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle dataset slug")
    ap.add_argument("--source", help="existing extracted dir (implies --skip-download)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--move", action="store_true",
                    help="move rather than copy (saves disk space)")
    args = ap.parse_args()

    download_dir = Path(args.source) if args.source else (config.RAW_DIR / "_download")

    if not (args.skip_download or args.source):
        download(args.slug, download_dir)
    if not download_dir.is_dir():
        raise SystemExit(f"Source directory not found: {download_dir}")

    print(f"Normalising {download_dir} -> {config.RAW_DIR} ...")
    counters = normalise(download_dir, args.move)
    total = sum(counters.values())
    if total == 0:
        raise SystemExit(
            "No images normalised. Inspect the extracted tree and, if the class "
            "folders use unexpected names, extend NEGATIVE_TOKENS / POSITIVE_TOKENS "
            "in this script (or use scripts/ingest_real_dataset.py for CSV layouts)."
        )

    remove_marker()
    print("\nIngested:")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name}: {counters[i]}")
    print(f"Total: {total} images")
    print("Synthetic marker removed.\n")
    print("Next:")
    print("  python -m src.preprocessing")
    print("  python -m src.train --model all")
    print("  python -m src.evaluate")


if __name__ == "__main__":
    main()
