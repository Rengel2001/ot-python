"""
VAE: Train a Variational Autoencoder on flattened MNIST, plot 2D latent space.

KL regularization pushes the latent distribution toward N(0,I), reducing but
not eliminating gaps between digit clusters.

References:
  - Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

log = logging.getLogger(__name__)


class _VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar


def run(X, y, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256

    model = _VAE(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    ds = TensorDataset(torch.tensor(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(200):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)
            mse = nn.MSELoss()(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + 0.001 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            log.info("VAE Epoch %3d | Loss: %.5f", epoch + 1, loss.item())

    model.eval()
    Z = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size]).to(device)
            mu, _ = model.encode(xb)
            Z.append(mu.cpu().numpy())
    Z = np.concatenate(Z)
    log.info("VAE embedding: %s", Z.shape)

    from src.visualization import plot_latent
    plot_latent(Z, y, save_path=output_dir / "vae-2d.png",
                title="VAE Latent Space")
