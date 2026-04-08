"""
AE-FM method: Flow Matching on 100-dim AE latent codes.

Trains a velocity field on conditional OT paths (data -> noise), generates by
sampling noise and integrating the reverse ODE. Decoded with the shared AE.

References:
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from core.training import LR_CONFIGS, EVAL_EVERY, PATIENCE, save_loss_plot

log = logging.getLogger(__name__)

NAME = "fm"
MAX_EPOCHS = 2000
BATCH_SIZE = 512
N_STEPS = 200    # ODE integration steps


class _VelocityNet(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def generate(Z, latent_dim, device, output_dir, dataset_config):
    n_gen = dataset_config.num_gen

    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0) + 1e-8
    Z_norm = (Z - z_mean) / z_std
    train_data = torch.tensor(Z_norm, dtype=torch.float32, device=device)
    n_train = len(Z)

    log.info("FM: max_epochs=%d, n_steps=%d, LR configs=%s",
             MAX_EPOCHS, N_STEPS, LR_CONFIGS)

    lr_histories = {}
    best_lr, best_eval_loss, best_state = None, float('inf'), None

    for lr in LR_CONFIGS:
        model = _VelocityNet(dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        eval_best, patience_ctr = float('inf'), 0
        accum_loss, accum_count = 0.0, 0

        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad()
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            x0 = train_data[idx]
            x1 = torch.randn_like(x0)
            t = torch.rand(BATCH_SIZE, 1, device=device)
            xt = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            pred_v = model(xt, t)
            loss = nn.MSELoss()(pred_v, target_v)
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_count += 1

            if (epoch + 1) % EVAL_EVERY == 0:
                avg_loss = accum_loss / accum_count
                history.append((epoch + 1, avg_loss))
                log.info("  [lr=%s] Epoch %4d | MSE: %.6f", lr, epoch + 1, avg_loss)
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

    log.info("  Best LR: %s (MSE: %.6f)", best_lr, best_eval_loss)
    save_loss_plot(lr_histories, output_dir, NAME, best_lr)

    model.load_state_dict(best_state)
    model.eval()
    dt = 1.0 / N_STEPS
    xt = torch.randn(n_gen, latent_dim, device=device)

    with torch.no_grad():
        for step in range(N_STEPS):
            t_val = 1.0 - step * dt
            t = torch.full((n_gen, 1), t_val, device=device)
            v = model(xt, t)
            xt = xt - v * dt

    gen_norm = xt.cpu().numpy()
    gen_codes = gen_norm * z_std + z_mean
    log.info("FM generated %d latent codes", n_gen)
    return gen_codes
