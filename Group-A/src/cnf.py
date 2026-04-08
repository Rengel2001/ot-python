"""
CNF: Map UMAP embeddings to [0,1]^2 via Continuous Normalizing Flow.

Trains via maximum likelihood with Hutchinson trace estimator for the
instantaneous change of variables. Diffeomorphic — gaps persist.

References:
  - Grathwohl et al., "FFJORD: Free-form Continuous Dynamics for Scalable
    Reversible Generative Models" (ICLR 2019)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as normal_dist

log = logging.getLogger(__name__)


class _CNFNet(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def _hutchinson_trace(model, x, t):
    x = x.detach().requires_grad_(True)
    f = model(x, t)
    eps = torch.randn_like(x)
    (ejp,) = torch.autograd.grad(f, x, grad_outputs=eps,
                                  create_graph=True, retain_graph=True)
    return (eps * ejp).sum(dim=1)


def run(Z_umap, y, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    train_steps = 20
    eval_steps = 100

    z_mean = Z_umap.mean(axis=0)
    z_std = Z_umap.std(axis=0)
    Z_norm = (Z_umap - z_mean) / z_std

    model = _CNFNet(dim=Z_umap.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = torch.tensor(Z_norm, dtype=torch.float32).to(device)
    n_train = len(Z_umap)

    dt = 1.0 / train_steps
    for epoch in range(200):
        optimizer.zero_grad()
        idx = torch.randint(0, n_train, (batch_size,))
        x = train_data[idx]

        log_det = torch.zeros(batch_size, device=device)
        xt = x.clone()
        for k in range(train_steps):
            t = torch.full((batch_size, 1), k * dt, device=device)
            trace = _hutchinson_trace(model, xt, t)
            v = model(xt.detach().requires_grad_(False), t)
            xt = xt + v * dt
            log_det = log_det + trace * dt

        log_p_base = -0.5 * (xt.pow(2).sum(dim=1) + Z_umap.shape[1] * np.log(2 * np.pi))
        loss = -(log_p_base + log_det).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            log.info("CNF Epoch %3d | NLL: %.4f", epoch + 1, loss.item())

    model.eval()
    dt = 1.0 / eval_steps
    xt = torch.tensor(Z_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        for step in range(eval_steps):
            t = torch.full((xt.shape[0], 1), step * dt, device=device)
            v = model(xt, t)
            xt = xt + v * dt

    z_gaussian = xt.cpu().numpy()
    Z_mapped = normal_dist.cdf(z_gaussian).astype(np.float32)

    from src.visualization import plot_latent
    plot_latent(Z_mapped, y, save_path=output_dir / "cnf-uniform.png",
                title="CNF -> [0,1]^2", xlim=(0, 1), ylim=(0, 1))
