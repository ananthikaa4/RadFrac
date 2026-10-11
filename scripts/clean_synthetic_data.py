"""Purge synthetic placeholder images from data/raw so they never mix with real data.

Why this exists as a dedicated script: a naive ``del synthetic_*.png`` is unsafe
in both directions.

* It can match **nothing** - earlier generator versions named files
  ``<class>_NNNN.png`` with no ``synthetic_`` prefix, so a glob on
  ``synthetic_*`` silently deletes zero files and leaves the training set
  contaminated while appearing to succeed.
* It can match **too much** - if real images were ever normalised with a
  colliding prefix, a broad glob would delete genuine data.

So detection is layered, most authoritative first:

1. **Marker file.** If ``data/raw/.SYNTHETIC_DATA`` exists, the generator wrote
   this folder, so every image under the class folders is synthetic.
2. **Filename pattern.** Otherwise, only files starting with the current
   ``synthetic_`` prefix, or matching the legacy ``<class>_NNNN.ext`` shape
   (4-digit counter, as the old generator produced), are removed.

Real images normalised by the ingest scripts use the ``xr_`` prefix and a 6-digit
counter, so they can never be caught by either rule.

Usage::

    python -m scripts.clean_synthetic_data            # dry run, shows what it would do
    python -m scripts.clean_synthetic_data --apply    # actually delete
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dataset_status import remove_marker  # noqa: E402

# Legacy generator naming: fractured_0000.png / not_fractured_0123.png
LEGACY_RE = re.compile(
    r"^(?:" + "|".join(re.escape(c) for c in config.CLASS_NAMES) + r")_\d{4}\.[a-z]+$",
    re.IGNORECASE,
)


def classify_files(raw_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]], bool]:
    """Return (synthetic_by_class, real_by_class, marker_present)."""
    marker = (raw_dir / config.SYNTHETIC_MARKER_FILE.name).exists()
    synthetic: dict[str, list[Path]] = {}
    real: dict[str, list[Path]] = {}

    for class_name in config.CLASS_NAMES:
        class_dir = raw_dir / class_name
        synthetic[class_name], real[class_name] = [], []
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in config.VALID_EXTENSIONS:
                continue
            name = p.name
            is_synth = (
                marker                                              # rule 1
                or name.startswith(config.SYNTHETIC_FILENAME_PREFIX)  # rule 2a
                or bool(LEGACY_RE.match(name))                        # rule 2b
            )
            # Never treat a properly-named real image as synthetic.
            if name.startswith(config.REAL_IMAGE_PREFIX):
                is_synth = False
            (synthetic if is_synth else real)[class_name].append(p)

    return synthetic, real, marker


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove synthetic placeholder images")
    ap.add_argument("--apply", action="store_true",
                    help="perform deletion (default is a dry run)")
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else config.RAW_DIR
    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    synthetic, real, marker = classify_files(raw_dir)
    total_synth = sum(len(v) for v in synthetic.values())
    total_real = sum(len(v) for v in real.values())

    print(f"raw dir: {raw_dir}")
    print(f"synthetic marker present: {marker}"
          + ("  -> every image here is generator output" if marker else ""))
    print()
    for class_name in config.CLASS_NAMES:
        print(f"  {class_name:<16} synthetic={len(synthetic[class_name]):<6} "
              f"real={len(real[class_name])}")
    print(f"  {'TOTAL':<16} synthetic={total_synth:<6} real={total_real}")

    if total_synth == 0:
        print("\nNothing to remove - no synthetic images detected.")
        # Still clear a stale marker so warnings do not persist on real data.
        if marker and total_real > 0:
            if args.apply:
                remove_marker(raw_dir)
                print("Stale synthetic marker removed.")
            else:
                print("Would remove the stale marker (re-run with --apply).")
        return

    if not args.apply:
        print(f"\nDRY RUN - would delete {total_synth} synthetic image(s).")
        for class_name in config.CLASS_NAMES:
            for p in synthetic[class_name][:3]:
                print(f"    would delete {p.name}")
            if len(synthetic[class_name]) > 3:
                print(f"    ... and {len(synthetic[class_name]) - 3} more in {class_name}")
        print("\nRe-run with --apply to delete.")
        return

    deleted = 0
    for class_name in config.CLASS_NAMES:
        for p in synthetic[class_name]:
            try:
                p.unlink()
                deleted += 1
            except OSError as exc:
                print(f"    could not delete {p}: {exc}")

    # No synthetic images left -> the marker must go, or every downstream
    # surface keeps showing the synthetic warning on real data.
    remove_marker(raw_dir)

    print(f"\nDeleted {deleted} synthetic image(s); synthetic marker cleared.")
    print("Remaining real images per class:")
    _, real_after, _ = classify_files(raw_dir)
    grand = 0
    for class_name in config.CLASS_NAMES:
        n = len(real_after[class_name])
        grand += n
        print(f"  {class_name:<16} {n}")
    print(f"  {'TOTAL':<16} {grand}")
    if grand == 0:
        print("\nWARNING: data/raw is now EMPTY. Download a real dataset before "
              "running src.preprocessing (see scripts/fetch_kaggle_dataset.py).")


if __name__ == "__main__":
    main()
