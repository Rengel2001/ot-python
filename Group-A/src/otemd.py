"""
OT-EMD: Map UMAP embeddings to [0,1]^2 via discrete Earth Mover's Distance.

Solves the exact discrete OT problem between UMAP points and uniform target
samples. Full coverage guaranteed — every target point is assigned.

References:
  - POT library: https://github.com/PythonOT/POT
"""

import logging
import time
import numpy as np
import ot

log = logging.getLogger(__name__)


def run(Z_umap, y, output_dir):
    n = len(Z_umap)

    z_min = Z_umap.min(axis=0)
    z_max = Z_umap.max(axis=0)
    Z_norm = (Z_umap - z_min) / (z_max - z_min)

    Z_target = np.random.RandomState(0).uniform(0, 1, size=(n, 2)).astype(np.float32)

    a = np.ones(n) / n
    b = np.ones(n) / n

    M = ot.dist(Z_norm, Z_target, metric="sqeuclidean")
    log.info("OT EMD: solving %dx%d transport problem...", n, n)
    start = time.time()
    T = ot.emd(a, b, M)
    log.info("OT EMD: solved in %.1fs", time.time() - start)

    assignment = T.argmax(axis=1)
    Z_mapped = Z_target[assignment]

    from src.visualization import plot_latent
    plot_latent(Z_mapped, y, save_path=output_dir / "otemd-uniform.png",
                title="OT-EMD -> [0,1]^2", xlim=(0, 1), ylim=(0, 1))
