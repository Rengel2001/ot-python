"""
OT-SWD: Map UMAP embeddings to [0,1]^2 via Sliced Wasserstein Distance.

Optimizes point positions to minimize SWD against uniform target samples.
Uses random 1D projections — an approximate OT method without full-coverage
guarantees.

References:
  - Bonneel et al., "Sliced and Radon Wasserstein Barycenters" (2015)
  - POT library: https://github.com/PythonOT/POT
"""

import logging
import time
import numpy as np
import torch

log = logging.getLogger(__name__)

N_PROJECTIONS = 200
LR = 0.01
EPOCHS = 100


def sliced_wasserstein_loss(X, Y, n_projections=N_PROJECTIONS):
    """Compute SWD between two 2D point sets via random 1D projections."""
    d = X.shape[1]
    theta = torch.randn(n_projections, d, device=X.device)
    theta = theta / theta.norm(dim=1, keepdim=True)

    proj_X = X @ theta.T  # (N, n_proj)
    proj_Y = Y @ theta.T  # (M, n_proj)

    proj_X_sorted, _ = proj_X.sort(dim=0)
    proj_Y_sorted, _ = proj_Y.sort(dim=0)

    return (proj_X_sorted - proj_Y_sorted).pow(2).mean()


def run(Z_umap, y, output_dir):
    n = len(Z_umap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_min = Z_umap.min(axis=0)
    z_max = Z_umap.max(axis=0)
    Z_norm = (Z_umap - z_min) / (z_max - z_min)

    Z_mapped = torch.tensor(Z_norm, dtype=torch.float32, device=device,
                            requires_grad=True)

    Z_target = torch.rand(n, 2, device=device)

    optimizer = torch.optim.Adam([Z_mapped], lr=LR)

    log.info("OT-SWD: optimizing %d points, %d projections, %d epochs...",
             n, N_PROJECTIONS, EPOCHS)
    start = time.time()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = sliced_wasserstein_loss(Z_mapped, Z_target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Z_mapped.clamp_(0, 1)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            log.info("  Epoch %3d/%d | SWD loss: %.6f", epoch + 1, EPOCHS,
                     loss.item())

    log.info("OT-SWD: done in %.1fs", time.time() - start)

    Z_out = Z_mapped.detach().cpu().numpy()

    from src.visualization import plot_latent
    plot_latent(Z_out, y, save_path=output_dir / "otswd-uniform.png",
                title="OT-SWD -> [0,1]^2", xlim=(0, 1), ylim=(0, 1))
