"""Generate a PDF report from the actual evaluation artifacts.

Builds ``report/RadFrac_report.pdf`` from ``outputs/metrics/comparison_metrics.json``
and the figures under ``outputs/figures/``. If the models were trained/evaluated
on synthetic data, a hard warning banner is rendered as the first thing on the
title page (and repeated on the results page) so the report can never be read as
presenting clinically meaningful performance.

Uses matplotlib's PdfPages, so no extra dependency is required.

Usage::

    python -m scripts.generate_report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dataset_status import WARNING_BODY, WARNING_TITLE  # noqa: E402


def _load_metrics() -> dict:
    if not config.COMPARISON_METRICS_FILE.exists():
        raise SystemExit(
            "No metrics found. Run 'python -m src.evaluate' first."
        )
    return json.loads(config.COMPARISON_METRICS_FILE.read_text())


def _draw_banner(ax) -> None:
    """Render the synthetic-data warning as a red boxed banner on an axis."""
    import textwrap

    from matplotlib.patches import Rectangle

    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
            facecolor="#ffe5e5", edgecolor="#c0392b", linewidth=2.5,
        )
    )
    ax.text(0.5, 0.80, "⚠️  SYNTHETIC DATA", transform=ax.transAxes,
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#c0392b")
    ax.text(0.5, 0.66, "these metrics are NOT clinically meaningful",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            color="#c0392b")
    body = "\n".join(textwrap.wrap(WARNING_BODY, width=64))
    ax.text(0.5, 0.36, body, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="#7b241c")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    metrics = _load_metrics()
    synthetic = bool(metrics.get("synthetic_data"))
    models = metrics.get("models", {})

    report_dir = config.PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = report_dir / "RadFrac_report.pdf"

    with PdfPages(out_pdf) as pdf:
        # ---- Title page ------------------------------------------------- #
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.92, "RadFrac", ha="center", fontsize=26, fontweight="bold")
        fig.text(0.5, 0.885, "Radiographic Fracture Classification — Model Comparison",
                 ha="center", fontsize=12)
        fig.text(0.5, 0.86, "Decision-support research project — not a medical device",
                 ha="center", fontsize=9, style="italic", color="#555555")
        if synthetic:
            ax = fig.add_axes([0.1, 0.55, 0.8, 0.25])
            _draw_banner(ax)
        # Summary text.
        best = metrics.get("best_model", "?")
        lines = [
            f"Classes: {', '.join(metrics.get('class_names', []))}",
            f"Selection metric: {metrics.get('selection_metric', 'roc_auc')}",
            f"Best model: {best}",
            f"Data type: {'SYNTHETIC (pipeline test only)' if synthetic else 'real / user-provided'}",
        ]
        fig.text(0.12, 0.48, "\n".join(lines), fontsize=10, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Results table page ----------------------------------------- #
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.95, "Results — held-out test set", ha="center",
                 fontsize=15, fontweight="bold")
        if synthetic:
            ax_b = fig.add_axes([0.1, 0.80, 0.8, 0.12])
            _draw_banner(ax_b)

        cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]
        table_rows = []
        for name, m in models.items():
            row = [name]
            for c in cols:
                v = m.get(c)
                row.append("n/a" if v is None else f"{v:.3f}")
            table_rows.append(row)
        ax_t = fig.add_axes([0.08, 0.55, 0.84, 0.2])
        ax_t.axis("off")
        if table_rows:
            table = ax_t.table(
                cellText=table_rows,
                colLabels=["model", *cols],
                loc="center", cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.6)

        # Embed ROC comparison if present.
        roc_path = config.FIGURES_DIR / "roc_comparison.png"
        if roc_path.exists():
            import matplotlib.image as mpimg
            ax_img = fig.add_axes([0.2, 0.08, 0.6, 0.4])
            ax_img.imshow(mpimg.imread(roc_path))
            ax_img.axis("off")
            ax_img.set_title("ROC curve comparison", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Confusion matrices page ------------------------------------ #
        cm_files = [(k, config.FIGURES_DIR / f"confusion_{k}.png") for k in models]
        cm_files = [(k, p) for k, p in cm_files if p.exists()]
        if cm_files:
            import matplotlib.image as mpimg
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.95, "Confusion matrices", ha="center",
                     fontsize=15, fontweight="bold")
            if synthetic:
                ax_b = fig.add_axes([0.1, 0.86, 0.8, 0.08])
                _draw_banner(ax_b)
            for i, (k, p) in enumerate(cm_files[:3]):
                ax = fig.add_axes([0.1, 0.58 - i * 0.28, 0.8, 0.24])
                ax.imshow(mpimg.imread(p))
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report written -> {out_pdf}"
          + ("  (SYNTHETIC banner included)" if synthetic else ""))


if __name__ == "__main__":
    main()
