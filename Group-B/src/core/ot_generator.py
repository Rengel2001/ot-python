"""
Algorithm 2: PL Extension (gen_P from pyOMT demo2.py).

Generates new latent codes by interpolating between neighboring OT cells.
Uses dihedral angles via paraboloid normal lifting to identify cell boundaries.

References:
  - An, Guo, Lei, Luo, Yau, Gu. "AE-OT" (ICLR 2020)
  - pyOMT: https://github.com/k2cu8/pyOMT
"""

import logging
import numpy as np
import torch

log = logging.getLogger(__name__)

# Generation defaults (matching pyOMT)
NUM_X = 20000
TOPK = 20
REC_GEN_DISTANCE = 0.75
BAT_SIZE_N = 1000


def generate_latent_codes(P, h, N, dim, bat_size_P, device,
                          n_gen=10000, topk=TOPK,
                          angle_thresh=1.1, dissim=REC_GEN_DISTANCE):
    bat_size_n = BAT_SIZE_N
    numX = NUM_X
    num_bat_x = numX // bat_size_n

    log.info("  Algorithm 2 (gen_P): numX=%d, topk=%d, thresh=%s, dissim=%s",
             numX, topk, angle_thresh, dissim)

    # Find topk neighbors for each MC sample
    I_all = -torch.ones(topk, numX, dtype=torch.long)
    qrng = torch.quasirandom.SobolEngine(dimension=dim)

    for ii in range(max(num_bat_x, 1)):
        samples = qrng.draw(bat_size_n).to(device) - 0.5
        topk_vals = torch.full((topk, bat_size_n), -1e30, device=device)
        topk_inds = torch.full((topk, bat_size_n), -1, dtype=torch.long, device=device)

        num_bat_P = N // bat_size_P
        for ip in range(num_bat_P):
            temp_P = P[ip * bat_size_P:(ip + 1) * bat_size_P]
            temp_h = h[ip * bat_size_P:(ip + 1) * bat_size_P]
            U = temp_P @ samples.t() + temp_h.unsqueeze(1)
            k_this = min(topk, bat_size_P)
            batch_vals, batch_inds = torch.topk(U, k_this, dim=0)
            batch_inds += ip * bat_size_P
            combined_vals = torch.cat([topk_vals, batch_vals], dim=0)
            combined_inds = torch.cat([topk_inds, batch_inds], dim=0)
            _, merge_idx = torch.topk(combined_vals, topk, dim=0)
            topk_vals = combined_vals.gather(0, merge_idx)
            topk_inds = combined_inds.gather(0, merge_idx)

        start = ii * bat_size_n
        end = min(start + bat_size_n, numX)
        for k in range(topk):
            I_all[k, start:end] = topk_inds[k, :end - start].cpu()

    log.info("  Found topk=%d neighbors for %d MC samples", topk, numX)

    # Create pairwise (primary, neighbor_k) combinations
    I_pairs = -torch.ones(2, (topk - 1) * numX, dtype=torch.long)
    for k in range(topk - 1):
        I_pairs[0, k * numX:(k + 1) * numX] = I_all[0, :]
        I_pairs[1, k * numX:(k + 1) * numX] = I_all[k + 1, :]

    valid_mask = (I_pairs[0] >= 0) & (I_pairs[1] >= 0)
    I_pairs = I_pairs[:, valid_mask]
    log.info("  %d valid pairs", I_pairs.shape[1])

    # Compute dihedral angles via paraboloid normal lifting
    P_cpu = P.cpu()
    nm = torch.cat([P_cpu, -torch.ones(N, 1)], dim=1)
    nm = nm / torch.norm(nm, dim=1, keepdim=True)

    cs = torch.sum(nm[I_pairs[0]] * nm[I_pairs[1]], dim=1)
    cs = torch.clamp(cs, max=1.0)
    theta = torch.acos(cs)

    log.info("  Dihedral angles: min=%.4f, max=%.4f, mean=%.4f",
             theta.min().item(), theta.max().item(), theta.mean().item())

    # Filter by angle threshold
    valid = theta <= angle_thresh
    I_gen = I_pairs[:, valid]
    log.info("  After angle filter: %d pairs (%.1f%% kept)",
             I_gen.shape[1], valid.float().mean().item() * 100)

    # Deduplicate by primary cell
    I_gen, _ = torch.sort(I_gen, dim=0)
    I_gen_np = I_gen.numpy()
    _, uni_idx = np.unique(I_gen_np[0], return_index=True)
    np.random.shuffle(uni_idx)
    I_gen = I_gen[:, torch.from_numpy(uni_idx)]

    numGen = I_gen.shape[1]
    log.info("  After dedup: %d unique pairs", numGen)

    # Interpolate
    rand_w = dissim * torch.ones(numGen, 1)
    P_gen = (1 - rand_w) * P_cpu[I_gen[0]] + rand_w * P_cpu[I_gen[1]]
    P_recon = P_cpu[I_gen[0]]
    P_all = torch.cat([P_gen, P_recon], dim=0)

    result = P_all.numpy()
    log.info("  Generated: %d interp + %d recon = %d total", numGen, numGen, len(result))
    return result
