"""
Autoencoder: Train a simple MLP AE on flattened MNIST, plot 2D latent space.

Learns a 2D bottleneck embedding without any density model. The resulting
latent space typically shows separated digit clusters with gaps between them.
Returns the embedding for reuse by aeot.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

log = logging.getLogger(__name__)


class _Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def run(X, y, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _Autoencoder(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    for epoch in range(200):
        model.train()
        total_loss = 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = nn.MSELoss()(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            avg = total_loss / len(loader)
            log.info("AE Epoch %3d | MSE: %.6f", epoch + 1, avg)

    model.eval()
    with torch.no_grad():
        Z = model.encoder(torch.tensor(X, dtype=torch.float32, device=device))
    Z = Z.cpu().numpy()
    log.info("AE embedding: %s", Z.shape)

    from src.visualization import plot_latent
    plot_latent(Z, y, save_path=output_dir / "ae-2d.png",
                title="AE Latent Space")
    return Z
