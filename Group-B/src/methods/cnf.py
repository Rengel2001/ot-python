"""
AE-CNF method: Continuous Normalizing Flow on 100-dim AE latent codes.

Trains via maximum likelihood with Hutchinson trace estimator for the
instantaneous change of variables. Generates by reverse ODE from Gaussian.
Decoded with the shared AE.

References:
  - Grathwohl et al., "FFJORD" (ICLR 2019)
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from core.training import LR_CONFIGS, EVAL_EVERY, PATIENCE, save_loss_plot

log = logging.getLogger(__name__)

NAME = "cnf"
MAX_EPOCHS = 1000
BATCH_SIZE = 256
TRAIN_STEPS = 10     # Euler steps during training
EVAL_STEPS = 200     # Euler steps during inference


class _CNFNet(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def _hutchinson_trace(model, x, t, n_hutchinson=2):
    x = x.detach().requires_grad_(True)
    f = model(x, t)
    trace = torch.zeros(x.shape[0], device=x.device)
    for _ in range(n_hutchinson):
        eps = torch.randn_like(x)
        (ejp,) = torch.autograd.grad(f, x, grad_outputs=eps,
                                      create_graph=True, retain_graph=True)
        trace = trace + (eps * ejp).sum(dim=1)
    return trace / n_hutchinson


def generate(Z, latent_dim, device, output_dir, dataset_config):
    n_gen = dataset_config.num_gen

    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0) + 1e-8
    Z_norm = (Z - z_mean) / z_std
    train_data = torch.tensor(Z_norm, dtype=torch.float32, device=device)
    n_train = len(Z)

    log.info("CNF: max_epochs=%d, train_steps=%d, eval_steps=%d, LR configs=%s",
             MAX_EPOCHS, TRAIN_STEPS, EVAL_STEPS, LR_CONFIGS)

    lr_histories = {}
    best_lr, best_eval_loss, best_state = None, float('inf'), None

    for lr in LR_CONFIGS:
        model = _CNFNet(dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        eval_best, patience_ctr = float('inf'), 0
        accum_loss, accum_count = 0.0, 0
        dt = 1.0 / TRAIN_STEPS

        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad()
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            x = train_data[idx]

            log_det = torch.zeros(BATCH_SIZE, device=device)
            xt = x.clone()
            for k in range(TRAIN_STEPS):
                t = torch.full((BATCH_SIZE, 1), k * dt, device=device)
                trace = _hutchinson_trace(model, xt, t)
                v = model(xt.detach().requires_grad_(False), t)
                xt = xt + v * dt
                log_det = log_det + trace * dt

            log_p_base = -0.5 * (xt.pow(2).sum(dim=1) + latent_dim * np.log(2 * np.pi))
            loss = -(log_p_base + log_det).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            accum_loss += loss.item()
            accum_count += 1

            if (epoch + 1) % EVAL_EVERY == 0:
                avg_loss = accum_loss / accum_count
                history.append((epoch + 1, avg_loss))
                log.info("  [lr=%s] Epoch %4d | NLL: %.4f", lr, epoch + 1, avg_loss)
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

    log.info("  Best LR: %s (NLL: %.4f)", best_lr, best_eval_loss)
    save_loss_plot(lr_histories, output_dir, NAME, best_lr)

    model.load_state_dict(best_state)
    model.eval()
    dt = 1.0 / EVAL_STEPS
    xt = torch.randn(n_gen, latent_dim, device=device)

    with torch.no_grad():
        for step in range(EVAL_STEPS):
            t_val = 1.0 - step * dt
            t = torch.full((n_gen, 1), t_val, device=device)
            v = model(xt, t)
            xt = xt - v * dt

    gen_norm = xt.cpu().numpy()
    gen_codes = gen_norm * z_std + z_mean
    log.info("CNF generated %d latent codes", n_gen)
    return gen_codes
