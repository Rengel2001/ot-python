"""
Shared visualization for Group A experiments.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

log = logging.getLogger(__name__)

# Updated Coloring to enhance visualization contrast
_rainbow = cm.get_cmap("rainbow")
DIGIT_COLORS = [_rainbow(i / 9) for i in range(10)]
DIGIT_COLORS[1] = (0.15, 0.15, 0.75, 1.0)
DIGIT_COLORS[2] = (0.60, 0.68, 0.85, 1.0)
DIGIT_COLORS[3] = (0.00, 0.75, 0.95, 1.0)
DIGIT_COLORS[4] = (0.08, 0.90, 0.13, 1.0)
DIGIT_COLORS[6] = (0.95, 0.90, 0.20, 1.0)
DIGIT_COLORS[7] = (1.00, 0.50, 0.00, 1.0)
DIGIT_COLORS[8] = (0.70, 0.10, 0.10, 1.0)
DIGIT_COLORS[9] = (1.00, 0.08, 0.58, 1.0)


def _legend_handles():
    return [plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=DIGIT_COLORS[d],
                       markersize=10, label=str(d)) for d in range(10)]


def plot_latent(Z, labels, save_path, title="Latent Space", xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("#f5f5f5")

    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[DIGIT_COLORS[digit]], s=8, alpha=0.5,
            label=str(digit), edgecolors="none", rasterized=True,
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(handles=_legend_handles(), title="Digit", loc="center left",
              bbox_to_anchor=(1.02, 0.5), framealpha=0.9, fontsize=9, title_fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot to %s", save_path)


def plot_power_diagram(P, h, labels, save_path, title="SDOT -> [0,1]^2",
                       resolution=500):
    pnorms = np.sum(P ** 2, axis=1)

    grid_x = np.linspace(0, 1, resolution)
    grid_y = np.linspace(0, 1, resolution)
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    chunk_size = 2000
    cell_ids = np.zeros(len(grid_points), dtype=np.int64)
    for start in range(0, len(grid_points), chunk_size):
        end = min(start + chunk_size, len(grid_points))
        scores = 2.0 * grid_points[start:end] @ P.T - pnorms + h
        cell_ids[start:end] = scores.argmax(axis=1)

    digit_grid = labels[cell_ids].reshape(resolution, resolution)

    rgba_colors = [mcolors.to_rgba(c) for c in DIGIT_COLORS]
    cmap = mcolors.ListedColormap(rgba_colors)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("#f5f5f5")
    ax.imshow(digit_grid, origin="lower", extent=[0, 1, 0, 1],
              cmap=cmap, vmin=0, vmax=9, interpolation="nearest", aspect="auto")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(handles=_legend_handles(), title="Digit", loc="center left",
              bbox_to_anchor=(1.02, 0.5), framealpha=0.9, fontsize=9, title_fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved power diagram to %s", save_path)
