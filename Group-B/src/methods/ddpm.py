"""
AE-DDPM method: Denoising Diffusion on 100-dim AE latent codes.

Trains a denoising model on latent codes. Generates via reverse diffusion
(noise -> latent codes). Decoded with the shared AE.

References:
  - Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from core.training import LR_CONFIGS, EVAL_EVERY, PATIENCE, save_loss_plot

log = logging.getLogger(__name__)

NAME = "ddpm"
MAX_EPOCHS = 5000
T_STEPS = 1000
BATCH_SIZE = 512


class _DenoiseNet(nn.Module):
    def __init__(self, dim, hidden=512, T_steps=1000):
        super().__init__()
        self.T_steps = T_steps
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
        t_emb = (t.float() / self.T_steps).unsqueeze(1)
        return self.net(torch.cat([x, t_emb], dim=1))


def generate(Z, latent_dim, device, output_dir, dataset_config):
    n_gen = dataset_config.num_gen

    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0) + 1e-8
    Z_norm = (Z - z_mean) / z_std

    betas = torch.linspace(1e-4, 0.02, T_STEPS, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    train_data = torch.tensor(Z_norm, dtype=torch.float32, device=device)
    n_train = len(Z)

    log.info("DDPM: T=%d, max_epochs=%d, LR configs=%s",
             T_STEPS, MAX_EPOCHS, LR_CONFIGS)

    lr_histories = {}
    best_lr, best_eval_loss, best_state = None, float('inf'), None

    for lr in LR_CONFIGS:
        model = _DenoiseNet(dim=latent_dim, T_steps=T_STEPS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        eval_best, patience_ctr = float('inf'), 0
        accum_loss, accum_count = 0.0, 0

        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad()
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            x0 = train_data[idx]
            t = torch.randint(0, T_STEPS, (BATCH_SIZE,), device=device)
            noise = torch.randn_like(x0)
            alpha_t = alphas_cumprod[t].unsqueeze(1)
            xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
            pred_noise = model(xt, t)
            loss = nn.MSELoss()(pred_noise, noise)
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_count += 1

            if (epoch + 1) % EVAL_EVERY == 0:
                avg_loss = accum_loss / accum_count
                history.append((epoch + 1, avg_loss))
                log.info("  [lr=%s] Epoch %5d | MSE: %.6f", lr, epoch + 1, avg_loss)
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
    xt = torch.randn(n_gen, latent_dim, device=device)

    with torch.no_grad():
        for t_val in reversed(range(T_STEPS)):
            t = torch.full((n_gen,), t_val, device=device, dtype=torch.long)
            pred_noise = model(xt, t)

            beta_t = betas[t_val]
            alpha_t = alphas[t_val]
            alpha_bar_t = alphas_cumprod[t_val]

            mean = (1 / torch.sqrt(alpha_t)) * (
                xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)

            if t_val > 0:
                noise = torch.randn_like(xt)
                xt = mean + torch.sqrt(beta_t) * noise
            else:
                xt = mean

    gen_norm = xt.cpu().numpy()
    gen_codes = gen_norm * z_std + z_mean
    log.info("DDPM generated %d latent codes", n_gen)
    return gen_codes
