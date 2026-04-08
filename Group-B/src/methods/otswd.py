"""
AE-OT-SWD method: Sliced Wasserstein Distance on 100-dim AE latent codes.

Trains a small network to map uniform samples to the latent code distribution
by minimizing the Sliced Wasserstein Distance. Generates by sampling uniform
and pushing through the learned map. Decoded with the shared AE.

References:
  - Bonneel et al., "Sliced and Radon Wasserstein Barycenters" (2015)
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from core.training import LR_CONFIGS, EVAL_EVERY, PATIENCE, save_loss_plot

log = logging.getLogger(__name__)

NAME = "otswd"
MAX_EPOCHS = 2000
N_PROJECTIONS = 500
BATCH_SIZE = 2048


class _TransportNet(nn.Module):
    """Maps uniform samples to latent code distribution."""
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def sliced_wasserstein_loss(X, Y, n_projections=N_PROJECTIONS):
    """Compute SWD between two point sets via random 1D projections."""
    d = X.shape[1]
    theta = torch.randn(n_projections, d, device=X.device)
    theta = theta / theta.norm(dim=1, keepdim=True)

    proj_X = X @ theta.T
    proj_Y = Y @ theta.T

    proj_X_sorted, _ = proj_X.sort(dim=0)
    proj_Y_sorted, _ = proj_Y.sort(dim=0)

    return (proj_X_sorted - proj_Y_sorted).pow(2).mean()


def generate(Z, latent_dim, device, output_dir, dataset_config):
    n_gen = dataset_config.num_gen

    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0) + 1e-8
    Z_norm = (Z - z_mean) / z_std
    train_data = torch.tensor(Z_norm, dtype=torch.float32, device=device)
    n_train = len(Z)

    log.info("OT-SWD: max_epochs=%d, projections=%d, batch=%d, LR configs=%s",
             MAX_EPOCHS, N_PROJECTIONS, BATCH_SIZE, LR_CONFIGS)

    lr_histories = {}
    best_lr, best_eval_loss, best_state = None, float('inf'), None

    for lr in LR_CONFIGS:
        model = _TransportNet(dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        eval_best, patience_ctr = float('inf'), 0
        accum_loss, accum_count = 0.0, 0

        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad()
            z_uniform = torch.randn(BATCH_SIZE, latent_dim, device=device)
            gen = model(z_uniform)
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            real = train_data[idx]
            loss = sliced_wasserstein_loss(gen, real)
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_count += 1

            if (epoch + 1) % EVAL_EVERY == 0:
                avg_loss = accum_loss / accum_count
                history.append((epoch + 1, avg_loss))
                log.info("  [lr=%s] Epoch %4d | SWD: %.6f", lr, epoch + 1, avg_loss)
                accum_loss, accum_count = 0.0, 0
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

    log.info("  Best LR: %s (SWD: %.6f)", best_lr, best_eval_loss)
    save_loss_plot(lr_histories, output_dir, NAME, best_lr)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_uniform = torch.randn(n_gen, latent_dim, device=device)
        gen_norm = model(z_uniform).cpu().numpy()

    gen_codes = gen_norm * z_std + z_mean
    log.info("OT-SWD generated %d latent codes", n_gen)
    return gen_codes
