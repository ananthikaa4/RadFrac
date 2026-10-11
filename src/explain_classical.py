"""Feature visualisation for the HOG + Canny + SVM member.

The Grad-CAM heatmap belongs to MobileNetV2 and says nothing about the SVM, so
the classical model needs its own honest visualisation. What it actually sees is:

* a **HOG** descriptor - local gradient-orientation histograms, rendered here as
  the conventional per-cell "spoke" diagram, and
* a **Canny** edge map plus the 4x4 grid of edge densities that are appended to
  the HOG vector.

These are the literal inputs to the classifier, not a saliency estimate, so they
are a faithful (if lower-level) explanation: they show the evidence available to
the SVM rather than guessing which part of it mattered.
"""
from __future__ import annotations

import numpy as np

from . import config


def hog_visualisation(img_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (hog_render, canny_edges) as float arrays in [0, 1]."""
    from skimage import exposure
    from skimage.feature import canny, hog

    gray = (img_uint8.astype("float32").mean(axis=-1) / 255.0
            if img_uint8.ndim == 3 else img_uint8.astype("float32") / 255.0)

    _, hog_img = hog(
        gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    # Stretch contrast so the faint spokes are actually visible.
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, max(hog_img.max(), 1e-6)))
    edges = canny(gray, sigma=config.CANNY_SIGMA).astype("float32")
    return hog_img.astype("float32"), edges


def edge_density_grid(edges: np.ndarray, grid: int = 4) -> np.ndarray:
    """The 4x4 Canny edge-density block actually appended to the feature vector."""
    h, w = edges.shape[:2]
    ch, cw = h // grid, w // grid
    out = np.zeros((grid, grid), dtype="float32")
    for i in range(grid):
        for j in range(grid):
            out[i, j] = float(edges[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].mean())
    return out


def render_classical_explanation(img_uint8: np.ndarray, out_path) -> "object":
    """Write a 3-panel figure: HOG spokes, Canny edges, edge-density grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    hog_img, edges = hog_visualisation(img_uint8)
    density = edge_density_grid(edges)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), facecolor="#0b0f14")
    for ax in axes:
        ax.set_facecolor("#0b0f14")
        ax.axis("off")

    axes[0].imshow(hog_img, cmap="magma")
    axes[0].set_title("HOG gradient orientations", color="#e6edf3", fontsize=9)
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny edge map", color="#e6edf3", fontsize=9)
    im = axes[2].imshow(density, cmap="viridis")
    axes[2].set_title("Edge density (4x4 features)", color="#e6edf3", fontsize=9)
    for (i, j), v in np.ndenumerate(density):
        axes[2].text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white", fontsize=7)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
