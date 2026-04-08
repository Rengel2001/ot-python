"""
VAE baseline: Variational Autoencoder with its own encoder.

Unlike other Group B methods, VAE does not use the shared AE. It trains its own
encoder with KL regularization (reparameterization trick) using the same InfoGAN
architecture (Lucic et al. 2018). Generates by sampling from N(0, I) and decoding.

Uses KL warmup (linear annealing over first 100 epochs) and dimension-scaled KL
weight to prevent posterior collapse in high-dimensional latent spaces.

References:
  - Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)
  - Lucic et al., "Are GANs Created Equal?" (NeurIPS 2018)
    https://github.com/google/compare_gan
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.training import (AE_EPOCHS, AE_BATCH, set_seed,
                            LR_CONFIGS, EVAL_EVERY, PATIENCE, save_loss_plot)

log = logging.getLogger(__name__)

NAME = "vae"
MAX_EPOCHS = AE_EPOCHS
KL_WEIGHT = 0.01
KL_WARMUP_EPOCHS = 100


class VAE(nn.Module):
    def __init__(self, dim_z=100, dim_c=1, img_h=28, img_w=28):
        super().__init__()
        self.dim_z = dim_z
        self.h4 = img_h // 4
        self.w4 = img_w // 4

        self.enc_conv = nn.Sequential(
            nn.Conv2d(dim_c, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc_fc = nn.Sequential(
            nn.Linear(128 * self.h4 * self.w4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(1024, dim_z)
        self.fc_logvar = nn.Linear(1024, dim_z)

        self.dec_fc = nn.Sequential(
            nn.Linear(dim_z, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 128 * self.h4 * self.w4),
            nn.BatchNorm1d(128 * self.h4 * self.w4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, dim_c, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.enc_conv(x)
        x = x.view(x.size(0), -1)
        h = self.enc_fc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        z = z.view(z.size(0), -1)
        x = self.dec_fc(z)
        x = x.view(x.size(0), 128, self.h4, self.w4)
        x = self.dec_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def generate(Z, latent_dim, device, output_dir, dataset_config,
             X_images=None):
    n_gen = dataset_config.num_gen
    cfg = dataset_config

    ds = TensorDataset(torch.tensor(X_images))
    loader = DataLoader(ds, batch_size=AE_BATCH, shuffle=True)

    log.info("VAE: max_epochs=%d, kl_weight=%s, warmup=%d, LR configs=%s",
             MAX_EPOCHS, KL_WEIGHT, KL_WARMUP_EPOCHS, LR_CONFIGS)

    lr_histories = {}
    best_lr, best_eval_loss, best_state = None, float('inf'), None

    for lr in LR_CONFIGS:
        model = VAE(dim_z=latent_dim, dim_c=cfg.dim_c,
                     img_h=cfg.img_h, img_w=cfg.img_w).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        eval_best, patience_ctr = float('inf'), 0

        for epoch in range(MAX_EPOCHS):
            beta = KL_WEIGHT * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)
            model.train()
            total_loss = 0
            for (xb,) in loader:
                xb = xb.to(device)
                recon, mu, logvar = model(xb)
                mse = nn.MSELoss()(recon, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse + beta * kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)

            if (epoch + 1) % EVAL_EVERY == 0:
                history.append((epoch + 1, avg_loss))
                log.info("  [lr=%s] Epoch %3d/%d | Loss: %.6f | beta: %.4f",
                         lr, epoch + 1, MAX_EPOCHS, avg_loss, beta)
                if avg_loss < eval_best - 1e-7:
                    eval_best = avg_loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= PATIENCE:
                        log.info("  Early stop at epoch %d", epoch + 1)
                        break

        lr_histories[lr] = history
        if eval_best < best_eval_loss:
            best_lr = lr
            best_eval_loss = eval_best
            best_state = model.state_dict()

    log.info("  Best LR: %s (Loss: %.6f)", best_lr, best_eval_loss)
    save_loss_plot(lr_histories, output_dir, NAME, best_lr,
                   skip_epochs=KL_WARMUP_EPOCHS)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_gen, latent_dim, device=device)
        gen_images = model.decoder(z).cpu().numpy()

    torch.save(model.state_dict(), output_dir / "vae_model.pt")
    log.info("VAE generated %d images", n_gen)
    return gen_images
