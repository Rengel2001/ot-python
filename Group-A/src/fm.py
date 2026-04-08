"""
Flow Matching: Map UMAP embeddings to [0,1]^2 via learned velocity field.

Trains a velocity field on conditional OT paths (data -> uniform), then
integrates forward to map all points. Diffeomorphic — gaps persist.

References:
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

import logging
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class _VelocityNet(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def run(Z_umap, y, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    n_steps = 100

    z_min = Z_umap.min(axis=0)
    z_max = Z_umap.max(axis=0)
    Z_norm = (Z_umap - z_min) / (z_max - z_min)

    model = _VelocityNet(dim=Z_umap.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = torch.tensor(Z_norm, dtype=torch.float32).to(device)
    n_train = len(Z_umap)

    for epoch in range(200):
        optimizer.zero_grad()
        idx = torch.randint(0, n_train, (batch_size,))
        x0 = train_data[idx]
        x1 = torch.rand_like(x0).to(device)
        t = torch.rand(batch_size, 1).to(device)
        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0
        pred_v = model(xt, t)
        loss = nn.MSELoss()(pred_v, target_v)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            log.info("FM Epoch %3d | Loss: %.5f", epoch + 1, loss.item())

    model.eval()
    dt = 1.0 / n_steps
    xt = torch.tensor(Z_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        for step in range(n_steps):
            t = torch.full((xt.shape[0], 1), step * dt).to(device)
            v = model(xt, t)
            xt = xt + v * dt

    Z_mapped = xt.cpu().numpy()

    from src.visualization import plot_latent
    plot_latent(Z_mapped, y, save_path=output_dir / "fm-uniform.png",
                title="FM -> [0,1]^2", xlim=(0, 1), ylim=(0, 1))
