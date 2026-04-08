"""
Algorithm 1: MC Semi-Discrete OT Solver.

Finds optimal height vector h (Brenier potential) using Monte Carlo cell volume
estimation + manual Adam optimizer.

Key details (matching pyOMT: https://github.com/k2cu8/pyOMT):
  - Sobol quasi-random sampling on [-0.5, 0.5]^d
  - Manual Adam without bias correction, betas=(0.9, 0.999)
  - Center h after Adam update (not gradient before)
  - Multi-batch averaging with adaptive doubling on stall
  - No M_MAX cap

References:
  - An, Guo, Lei, Luo, Yau, Gu. "AE-OT" (ICLR 2020)
"""

import logging
import time
import torch

log = logging.getLogger(__name__)


# OT solver hyperparameters (matching pyOMT)
OT_LR = 0.05
OT_STEPS = 20000
GRAD_NORM_THRESHOLD = 2e-3
STALL_PATIENCE = 30
BAT_SIZE_N = 1000
INIT_NUM_BAT = 20
LOG_EVERY = 50


def cal_measure(samples, P, h, num_P, bat_size_P, bat_size_n, device):
    tot_ind = torch.full((bat_size_n,), -1, dtype=torch.long, device=device)
    tot_ind_val = torch.full((bat_size_n,), -1e30, dtype=torch.float, device=device)
    num_bat_P = num_P // bat_size_P

    for i in range(num_bat_P):
        temp_P = P[i * bat_size_P:(i + 1) * bat_size_P]
        temp_h = h[i * bat_size_P:(i + 1) * bat_size_P]
        U = temp_P @ samples.t() + temp_h.unsqueeze(1)
        ind_val, ind = torch.max(U, dim=0)
        ind += i * bat_size_P
        better = ind_val > tot_ind_val
        tot_ind_val[better] = ind_val[better]
        tot_ind[better] = ind[better]

    g = torch.bincount(tot_ind, minlength=num_P).float()
    g /= bat_size_n
    return g


def update_h_adam(h, g, adam_m, adam_v, num_P, lr):
    g -= 1.0 / num_P
    adam_m.mul_(0.9).add_(g, alpha=0.1)
    adam_v.mul_(0.999).add_(g * g, alpha=0.001)
    delta_h = -lr * adam_m / (torch.sqrt(adam_v) + 1e-8)
    h.add_(delta_h)
    h.sub_(h.mean())


def mc_semidiscrete_ot(source_points, dim, device, bat_size_P):
    N = len(source_points)
    bat_size_n = BAT_SIZE_N

    while N % bat_size_P != 0:
        bat_size_P -= 1
    log.info("  bat_size_P adjusted to %d (divides N=%d)", bat_size_P, N)

    P = torch.tensor(source_points, dtype=torch.float32, device=device)
    h = torch.zeros(N, dtype=torch.float32, device=device)
    adam_m = torch.zeros(N, dtype=torch.float32, device=device)
    adam_v = torch.zeros(N, dtype=torch.float32, device=device)

    qrng = torch.quasirandom.SobolEngine(dimension=dim)
    dyn_num_bat = INIT_NUM_BAT
    total_mc = bat_size_n * dyn_num_bat

    log.info("  Sobol on [-0.5, 0.5]^%d, bat_size_n=%d, init_num_bat=%d, N=%d, lr=%s",
             dim, bat_size_n, dyn_num_bat, N, OT_LR)

    curr_best_g_norm = 1e20
    count_bad = 0
    history = {"step": [], "max_vol_error": [], "mean_vol_error": [],
               "empty_cells": [], "g_norm": [], "total_mc": [], "wall_time": []}
    start_time = time.time()

    log.info("  %6s | %9s | %9s | %6s | %9s | %7s",
             "step", "max_ratio", "g_norm", "empty", "total_mc", "time")
    log.info("  %s-+-%s-+-%s-+-%s-+-%s-+-%s",
             "-" * 6, "-" * 9, "-" * 9, "-" * 6, "-" * 9, "-" * 7)

    for step in range(OT_STEPS):
        qrng.reset()
        g_sum = torch.zeros(N, dtype=torch.float32, device=device)
        for _ in range(dyn_num_bat):
            samples = qrng.draw(bat_size_n).to(device)
            samples.add_(-0.5)
            g = cal_measure(samples, P, h, N, bat_size_P, bat_size_n, device)
            g_sum.add_(g)

        g_avg = g_sum / dyn_num_bat
        update_h_adam(h, g_avg, adam_m, adam_v, N, OT_LR)

        g_norm = torch.sqrt(torch.sum(g_avg * g_avg)).item()
        num_zero = torch.sum(g_avg == -1.0 / N).item()
        g_ratio = (torch.abs(g_avg).max() * N).item()

        if g_norm < GRAD_NORM_THRESHOLD:
            elapsed = time.time() - start_time
            total_mc = bat_size_n * dyn_num_bat
            log.info("  Converged at step %d: g_norm=%.6f < %s",
                     step + 1, g_norm, GRAD_NORM_THRESHOLD)
            history["step"].append(step + 1)
            history["max_vol_error"].append(g_ratio)
            history["mean_vol_error"].append(torch.abs(g_avg).mean().item())
            history["empty_cells"].append(int(num_zero))
            history["g_norm"].append(g_norm)
            history["total_mc"].append(total_mc)
            history["wall_time"].append(elapsed)
            break

        if g_norm <= curr_best_g_norm:
            curr_best_g_norm = g_norm
            count_bad = 0
        else:
            count_bad += 1
        if count_bad > STALL_PATIENCE:
            dyn_num_bat *= 2
            total_mc = bat_size_n * dyn_num_bat
            log.info("  [Step %d: doubling to %d batches, total MC = %d]",
                     step + 1, dyn_num_bat, total_mc)
            count_bad = 0
            curr_best_g_norm = 1e20

        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            total_mc = bat_size_n * dyn_num_bat
            history["step"].append(step + 1)
            history["max_vol_error"].append(g_ratio)
            history["mean_vol_error"].append(torch.abs(g_avg).mean().item())
            history["empty_cells"].append(int(num_zero))
            history["g_norm"].append(g_norm)
            history["total_mc"].append(total_mc)
            history["wall_time"].append(elapsed)
            log.info("  %6d | %9.3f | %9.6f | %6d | %9d | %6.1fs",
                     step + 1, g_ratio, g_norm, int(num_zero), total_mc, elapsed)

    return h.detach(), P, bat_size_P, history
