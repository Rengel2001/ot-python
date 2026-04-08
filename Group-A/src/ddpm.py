"""
DDPM: Map UMAP embeddings to [0,1]^2 via stochastic forward diffusion.

Applies closed-form forward diffusion (data -> noise), then CDF to get uniform.
Destroys manifold structure — spatial coherence is lost.

References:
  - Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as normal_dist

log = logging.getLogger(__name__)


class _DenoiseNet(nn.Module):
    def __init__(self, dim=2, T_steps=200):
        super().__init__()
        self.T_steps = T_steps
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )

    def forward(self, x, t):
        t_emb = (t.float() / self.T_steps).unsqueeze(1)
        return self.net(torch.cat([x, t_emb], dim=1))


def run(Z_umap, y, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T_steps = 200
    batch_size = 256

    z_mean = Z_umap.mean(axis=0)
    z_std = Z_umap.std(axis=0)
    Z_norm = (Z_umap - z_mean) / z_std

    betas = torch.linspace(1e-4, 0.02, T_steps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    model = _DenoiseNet(dim=Z_umap.shape[1], T_steps=T_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    train_data = torch.tensor(Z_norm, dtype=torch.float32).to(device)
    n_train = len(Z_umap)

    for epoch in range(500):
        optimizer.zero_grad()
        idx = torch.randint(0, n_train, (batch_size,))
        x0 = train_data[idx]
        t = torch.randint(0, T_steps, (batch_size,)).to(device)
        noise = torch.randn_like(x0)
        alpha_t = alphas_cumprod[t].unsqueeze(1)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        pred_noise = model(xt, t)
        loss = nn.MSELoss()(pred_noise, noise)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 200 == 0:
            log.info("DDPM Epoch %4d | Loss: %.5f", epoch + 1, loss.item())

    # Stochastic forward diffusion: data -> noise (closed-form)
    x0 = torch.tensor(Z_norm, dtype=torch.float32).to(device)
    alpha_bar_T = alphas_cumprod[-1]
    noise = torch.randn_like(x0)
    xT = torch.sqrt(alpha_bar_T) * x0 + torch.sqrt(1 - alpha_bar_T) * noise

    z_gaussian = xT.cpu().numpy()
    Z_mapped = normal_dist.cdf(z_gaussian).astype(np.float32)

    from src.visualization import plot_latent
    plot_latent(Z_mapped, y, save_path=output_dir / "ddpm-uniform.png",
                title="DDPM -> [0,1]^2", xlim=(0, 1), ylim=(0, 1))
