"""
UMAP: Embed MNIST (784-dim) to 2D using Uniform Manifold Approximation.

Produces the shared 2D embedding used by all mapping methods (NF, FM, CNF,
DDPM, OT-EMD) in Group A. Returns the embedding for reuse.

References:
  - McInnes et al., "UMAP: Uniform Manifold Approximation and Projection
    for Dimension Reduction" (2018), https://github.com/lmcinnes/umap
"""

import logging
import umap

log = logging.getLogger(__name__)


def run(X, y, output_dir):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=True,
    )
    Z = reducer.fit_transform(X)
    log.info("UMAP embedding: %s", Z.shape)

    from src.visualization import plot_latent
    plot_latent(Z, y, save_path=output_dir / "umap-2d.png",
                title="UMAP Latent Space")
    return Z
