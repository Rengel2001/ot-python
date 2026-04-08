"""
AE-NF method: Normalizing Flow (RealNVP) on 100-dim AE latent codes.

Trains an invertible flow (latent codes -> Gaussian), generates by sampling
Gaussian and inverting. Decoded with the shared AE.

References:
  - Dinh et al., "Density estimation using Real-NVP" (ICLR 2017)
  - normflows library: https://github.com/VincentStimper/normalizing-flows
"""

import logging
import numpy as np
import torch
import normflows as nf_lib

from core.training import LR_CONFIGS, EVAL_EVERY, PATIENCE, save_loss_plot

log = logging.getLogger(__name__)

NAME = "nf"
MAX_EPOCHS = 500
K = 16           # number of coupling layers
HIDDEN = 256     # hidden units per MLP
BATCH_SIZE = 512


def generate(Z, latent_dim, device, output_dir, dataset_config):
    n_gen = dataset_config.num_gen

    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0) + 1e-8
    Z_norm = (Z - z_mean) / z_std
    train_tensor = torch.tensor(Z_norm, dtype=torch.float32, device=device)
    n_train = len(Z)

    log.info("NF: K=%d, hidden=%d, max_epochs=%d, LR configs=%s",
             K, HIDDEN, MAX_EPOCHS, LR_CONFIGS)

    lr_histories = {}
    best_lr, best_eval_loss, best_state = None, float('inf'), None

    for lr in LR_CONFIGS:
        flows = []
        for i in range(K):
            mask = torch.zeros(latent_dim)
            mask[i % 2::2] = 1.0
            s = nf_lib.nets.MLP([latent_dim, HIDDEN, HIDDEN, latent_dim], init_zeros=True)
            t = nf_lib.nets.MLP([latent_dim, HIDDEN, HIDDEN, latent_dim], init_zeros=True)
            flows.append(nf_lib.flows.MaskedAffineFlow(mask, t, s))
        base = nf_lib.distributions.DiagGaussian(latent_dim)
        model = nf_lib.NormalizingFlow(base, flows).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        eval_best, patience_ctr = float('inf'), 0
        accum_loss, accum_count = 0.0, 0

        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad()
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            loss = model.forward_kld(train_tensor[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    with torch.no_grad():
        z_sample, _ = model.sample(n_gen)
        gen_norm = z_sample.cpu().numpy()

    gen_codes = gen_norm * z_std + z_mean
    log.info("NF generated %d latent codes", n_gen)
    return gen_codes
