"""Single source of truth for whether the training data is synthetic.

Every surface that presents model results (the evaluation CLI, the Streamlit
demo, the PDF report) calls :func:`is_synthetic_data` and, when true, shows the
same hard warning so synthetic-data metrics can never be mistaken for real,
clinically meaningful performance.

Detection is deliberately layered so it is hard to defeat by accident:

1. A marker file (``.SYNTHETIC_DATA``) written by the synthetic generator.
   This is recorded into the split manifests too, so it survives even if the
   raw data is later cleared.
2. A filename-prefix heuristic over the raw class folders
   (``fractured_*`` / ``not_fractured_*`` as produced by the generator).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config

WARNING_TITLE = "!! SYNTHETIC DATA -- these metrics are NOT clinically meaningful"
WARNING_BODY = (
    "The models were trained/evaluated on synthetically generated images "
    "(scripts/make_synthetic_data.py), which are an easy, artificial task used "
    "only to verify the pipeline runs end-to-end. Replace data/raw/ with a real "
    "radiograph dataset and re-run preprocessing/training/evaluation before "
    "drawing any conclusions."
)


def _marker_present(raw_dir: Path) -> bool:
    return (raw_dir / config.SYNTHETIC_MARKER_FILE.name).exists()


def _filenames_look_synthetic(raw_dir: Path) -> bool:
    """True if the raw class folders are dominated by generator-named files."""
    total = 0
    synthetic = 0
    for class_name in config.CLASS_NAMES:
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() not in config.VALID_EXTENSIONS:
                continue
            total += 1
            if p.name.startswith(config.SYNTHETIC_FILENAME_PREFIXES):
                synthetic += 1
    # Require a clear majority so a handful of oddly-named real files don't trip
    # the heuristic, and so an all-synthetic folder is caught even without the
    # marker file.
    return total > 0 and synthetic / total >= 0.5


def is_synthetic_data(raw_dir: Path | str | None = None) -> bool:
    """Return True if the dataset under ``raw_dir`` appears to be synthetic."""
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    if _marker_present(raw_dir):
        return True
    return _filenames_look_synthetic(raw_dir)


def write_marker(raw_dir: Path | str | None = None, meta: dict | None = None) -> None:
    """Write the synthetic-data marker file (called by the generator)."""
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / config.SYNTHETIC_MARKER_FILE.name
    payload = {"synthetic": True, **(meta or {})}
    marker.write_text(json.dumps(payload, indent=2))


def remove_marker(raw_dir: Path | str | None = None) -> None:
    """Remove the marker (call this after replacing raw data with a real set)."""
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    marker = raw_dir / config.SYNTHETIC_MARKER_FILE.name
    if marker.exists():
        marker.unlink()


def banner(width: int = 74) -> str:
    """A plain-text (ASCII-only) warning banner for CLI output.

    Kept ASCII so it renders on any console encoding (e.g. Windows cp1252)
    without raising UnicodeEncodeError.
    """
    line = "=" * width
    import textwrap

    wrapped = "\n".join(textwrap.wrap(WARNING_BODY, width=width - 2))
    return f"\n{line}\n{WARNING_TITLE}\n{line}\n{wrapped}\n{line}\n"
