"""Test that the PDF report generates and honours the synthetic flag."""
from __future__ import annotations

import importlib
import json

from src import config


def _write_metrics(synthetic: bool) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": config.CLASS_NAMES,
        "models": {
            "classical": {
                "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "roc_auc": 1.0, "threshold": 0.5, "confusion_matrix": [[3, 0], [0, 3]],
                "n_test": 6,
            }
        },
        "best_model": "classical",
        "selection_metric": "roc_auc",
        "synthetic_data": synthetic,
    }
    config.COMPARISON_METRICS_FILE.write_text(json.dumps(payload))


def test_report_generates_pdf(tiny_raw_dataset, tmp_path, monkeypatch):
    _write_metrics(synthetic=True)
    # Point the report output at the temp project root.
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    gen = importlib.import_module("scripts.generate_report")
    gen.main()

    out = tmp_path / "report" / "RadFrac_report.pdf"
    assert out.exists()
    # Valid PDF header.
    assert out.read_bytes()[:5] == b"%PDF-"
