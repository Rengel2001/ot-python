"""
Normalizing Flow: Map UMAP embeddings to [0,1]^2 via RealNVP.

Trains an invertible flow (data -> Gaussian), then applies CDF to get uniform.
Diffeomorphic — preserves topology, so gaps in the UMAP embedding persist.

References:
  - Dinh et al., "Density estimation using Real-NVP" (ICLR 2017)
  - normflows library: https://github.com/VincentStimper/normalizing-flows
"""

import logging
import numpy as np
import torch
import normflows as nf
from scipy.stats import norm as normal_dist

log = logging.getLogger(__name__)


def run(Z_umap, y, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = Z_umap.shape[1]

    z_mean = Z_umap.mean(axis=0)
    z_std = Z_umap.std(axis=0)
    Z_norm = (Z_umap - z_mean) / z_std

    b = torch.tensor([1, 0])
    flows = []
    for i in range(8):
        s = nf.nets.MLP([latent_size, 64, 64, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 64, 64, latent_size], init_zeros=True)
        mask = b if i % 2 == 0 else 1 - b
        flows.append(nf.flows.MaskedAffineFlow(mask, t, s))

    base = nf.distributions.DiagGaussian(latent_size)
    model = nf.NormalizingFlow(base, flows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_tensor = torch.tensor(Z_norm, dtype=torch.float32).to(device)

    for epoch in range(150):
        optimizer.zero_grad()
        loss = model.forward_kld(train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            log.info("NF Epoch %3d | NLL: %.4f", epoch + 1, loss.item())

    model.eval()
    with torch.no_grad():
        z_normal, _ = model.inverse_and_log_det(train_tensor)
        z_normal = z_normal.cpu().numpy()

    Z_mapped = normal_dist.cdf(z_normal).astype(np.float32)

    from src.visualization import plot_latent
    plot_latent(Z_mapped, y, save_path=output_dir / "nf-uniform.png",
                title="NF -> [0,1]^2", xlim=(0, 1), ylim=(0, 1))
