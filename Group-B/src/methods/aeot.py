"""
AE-OT method: Algorithm 1 (OT solver) + Algorithm 2 (PL extension generation).

Uses the shared AE backbone. Finds the optimal transport map via MC semi-discrete
OT (Algorithm 1), then generates new latent codes by interpolating between
neighboring power diagram cells (Algorithm 2).

References:
  - An, Guo, Lei, Luo, Yau, Gu. "AE-OT" (ICLR 2020)
  - pyOMT: https://github.com/k2cu8/pyOMT
"""

import logging
import time
import torch

from core.ot_solver import mc_semidiscrete_ot
from core.ot_generator import generate_latent_codes
from core.training import save_convergence_plot

log = logging.getLogger(__name__)

NAME = "aeot"

ANGLE_THRESHOLDS = {
    "mnist": 0.5,
    "fashion_mnist": 0.9,
    "cifar10": 1.1,
    "celeba": 0.7,
}


def generate(Z, latent_dim, device, output_dir, dataset_config):
    bat_size_P = dataset_config.bat_size_P
    angle_thresh = ANGLE_THRESHOLDS[dataset_config.name]

    log.info("Running Algorithm 1 (OT solver)...")
    start = time.time()
    h, P, bat_size_P, history = mc_semidiscrete_ot(Z, dim=latent_dim,
                                                    device=device,
                                                    bat_size_P=bat_size_P)
    ot_time = time.time() - start
    log.info("OT done in %.1fs", ot_time)
    save_convergence_plot(history, output_dir)

    torch.save({'h': h.cpu().numpy(), 'P': P.cpu().numpy(), 'Z': Z,
                'bat_size_P': bat_size_P, 'latent_dim': latent_dim},
               output_dir / "ot_state.pt")

    N = len(Z)
    log.info("Running Algorithm 2 (generation, angle_thresh=%.1f)...", angle_thresh)
    codes = generate_latent_codes(P, h, N, latent_dim, bat_size_P,
                                  device, angle_thresh=angle_thresh)
    return codes
