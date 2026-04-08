"""
AE training, LR sweep, checkpoint management, and visualization utilities.

Matches pyOMT training procedure (https://github.com/k2cu8/pyOMT):
  Stage 1: MSE + L1 latent regularization (500 epochs)
  Stage 2: Freeze encoder, refine decoder only (500 epochs)

References:
  - An, Guo, Lei, Luo, Yau, Gu. "AE-OT" (ICLR 2020)
  - Lucic et al., "Are GANs Created Equal?" (NeurIPS 2018)
    https://github.com/google/compare_gan
"""

import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

SEED = 42
AE_EPOCHS = 500
AE_REFINE_EPOCHS = 500
AE_BATCH = 512
L1_WEIGHT = 1e-5

# --- LR Sweep + Early Stopping ---
LR_CONFIGS = [1e-4, 5e-4, 1e-3, 2e-3]
EVAL_EVERY = 10       # average and log loss every N epochs
PATIENCE = 30         # stop after PATIENCE eval points (~300 epochs) with no improvement


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- AE Training Loop ---

def train_autoencoder(model, X, device, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(X))
    loader = DataLoader(ds, batch_size=AE_BATCH, shuffle=True)
    latent_dim = model.dim_z
    history = []

    log.info("  Stage 1: Training AE (dim_z=%d, lr=%s, epochs=%d)...",
             latent_dim, lr, AE_EPOCHS)

    best_loss, patience_ctr, last_epoch = float('inf'), 0, 0
    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, z = model(xb)
            mse_loss = nn.MSELoss()(recon, xb)
            l1_loss = z.abs().sum() / z.shape[0]
            loss = mse_loss + L1_WEIGHT * l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += mse_loss.item()
        avg = total_loss / len(loader)
        last_epoch = epoch + 1
        if last_epoch % EVAL_EVERY == 0:
            history.append((last_epoch, avg))
            log.info("    Epoch %3d/%d | MSE: %.6f", last_epoch, AE_EPOCHS, avg)
            if avg < best_loss - 1e-7:
                best_loss = avg
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    log.info("    Early stop at epoch %d", last_epoch)
                    break

    log.info("\n  Stage 2: Refining decoder (encoder frozen, epochs=%d)...",
             AE_REFINE_EPOCHS)

    for param in model.enc_conv.parameters():
        param.requires_grad = False
    for param in model.enc_fc.parameters():
        param.requires_grad = False

    decoder_params = list(model.dec_fc.parameters()) + list(model.dec_conv.parameters())
    refine_optimizer = torch.optim.Adam(decoder_params, lr=lr)

    best_loss, patience_ctr = float('inf'), 0
    for epoch in range(AE_REFINE_EPOCHS):
        model.train()
        total_loss = 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = nn.MSELoss()(recon, xb)
            refine_optimizer.zero_grad()
            loss.backward()
            refine_optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if (epoch + 1) % EVAL_EVERY == 0:
            history.append((last_epoch + epoch + 1, avg))
            log.info("    Refine %3d/%d | MSE: %.6f", epoch + 1, AE_REFINE_EPOCHS, avg)
            if avg < best_loss - 1e-7:
                best_loss = avg
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    log.info("    Early stop at epoch %d", epoch + 1)
                    break

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    Z = []
    with torch.no_grad():
        for i in range(0, len(X), AE_BATCH):
            xb = torch.tensor(X[i:i+AE_BATCH]).to(device)
            z = model.encoder(xb)
            z = z.view(z.size(0), -1)
            Z.append(z.cpu().numpy())
    Z = np.concatenate(Z)
    log.info("  Latent stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
             Z.mean(), Z.std(), Z.min(), Z.max())

    return Z, model, history


# --- LR Sweep + Checkpoint ---

def ae_lr_sweep(cfg, device):
    from core.ae_architecture import Autoencoder
    from core.fid_computation import compute_recon_fid

    output_dir = Path("output/models") / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "ae_checkpoint.pt"
    if ckpt_path.exists():
        log.info("  Skipping %s AE — checkpoint exists", cfg.name)
        return

    fh = logging.FileHandler(output_dir / "train_log.txt", mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(fh)

    latent_dim = 100
    X_images = cfg.load_fn()
    N = len(X_images)

    log.info("\n" + "=" * 60)
    log.info("  AE LR Sweep | %s | N=%d | %dx%d", cfg.name, N, cfg.img_h, cfg.img_w)
    log.info("  latent_dim=%d | configs=%s", latent_dim, LR_CONFIGS)
    log.info("=" * 60)

    lr_histories = {}
    best_lr, best_final_loss = None, float('inf')
    best_Z, best_model_state = None, None

    for lr in LR_CONFIGS:
        log.info("\n--- LR = %s ---", lr)
        set_seed(SEED)
        model = Autoencoder(
            dim_z=latent_dim, dim_c=cfg.dim_c,
            img_h=cfg.img_h, img_w=cfg.img_w).to(device)

        Z, model, history = train_autoencoder(model, X_images, device, lr=lr)
        lr_histories[lr] = history
        final_loss = history[-1][1] if history else float('inf')
        log.info("  LR %s -> final MSE: %.6f", lr, final_loss)

        if final_loss < best_final_loss:
            best_lr = lr
            best_final_loss = final_loss
            best_Z = Z
            best_model_state = model.state_dict()

    log.info("\n  Best LR: %s (MSE: %.6f)", best_lr, best_final_loss)
    save_loss_plot(lr_histories, output_dir, "AE", best_lr)

    # Reload best model for FID and saving
    model = Autoencoder(
        dim_z=latent_dim, dim_c=cfg.dim_c,
        img_h=cfg.img_h, img_w=cfg.img_w).to(device)
    model.load_state_dict(best_model_state)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    recon_fid = compute_recon_fid(model, X_images, device, output_dir)
    log.info("  Recon FID: %.2f", recon_fid)
    save_recon_images(model, X_images, device, output_dir, n_channels=cfg.dim_c)

    torch.save({
        'model_state_dict': best_model_state,
        'Z': best_Z,
        'latent_dim': latent_dim,
        'best_lr': best_lr,
        'recon_fid': recon_fid,
        'dim_c': cfg.dim_c,
        'img_h': cfg.img_h,
        'img_w': cfg.img_w,
    }, ckpt_path)

    logging.getLogger().removeHandler(fh)
    fh.close()


def load_ae_checkpoint(dataset_name, device):
    from core.ae_architecture import Autoencoder

    ckpt_path = Path("output/models") / dataset_name / "ae_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No AE checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Autoencoder(
        dim_z=ckpt['latent_dim'], dim_c=ckpt['dim_c'],
        img_h=ckpt['img_h'], img_w=ckpt['img_w']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, ckpt['Z'], ckpt['latent_dim']


# --- Decode + Visualization ---

def decode_latent_codes(model, codes, latent_dim, device, batch_size=AE_BATCH):
    model.eval()
    images = []
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            z = torch.tensor(codes[i:i+batch_size], dtype=torch.float32, device=device)
            z = z.view(-1, latent_dim, 1, 1)
            imgs = model.decoder(z)
            images.append(imgs.cpu().numpy())
    return np.concatenate(images)


def save_sample_images(images, output_dir, title="Generated", n_channels=1):
    rows, cols = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    fig.suptitle(f"64 {title} Samples", fontsize=14)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < len(images):
                if n_channels == 1:
                    ax.imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(np.transpose(images[idx], (1, 2, 0)).clip(0, 1))
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_recon_images(model, X_images, device, output_dir, n=64, n_channels=1):
    model.eval()
    idx = np.random.choice(len(X_images), n, replace=False)
    xb = torch.tensor(X_images[idx]).to(device)
    with torch.no_grad():
        recon, _ = model(xb)

    rows = 8
    fig, axes = plt.subplots(rows, 16, figsize=(16, 8))
    fig.suptitle("Reconstructions (left=original, right=reconstructed)", fontsize=14)
    for i in range(rows):
        for j in range(8):
            idx_img = i * 8 + j
            if idx_img < n:
                orig = xb[idx_img].cpu().numpy()
                rec = recon[idx_img].cpu().numpy()
                if n_channels == 1:
                    axes[i][j * 2].imshow(orig[0], cmap="gray", vmin=0, vmax=1)
                    axes[i][j * 2 + 1].imshow(rec[0], cmap="gray", vmin=0, vmax=1)
                else:
                    axes[i][j * 2].imshow(np.transpose(orig, (1, 2, 0)).clip(0, 1))
                    axes[i][j * 2 + 1].imshow(np.transpose(rec, (1, 2, 0)).clip(0, 1))
            axes[i][j * 2].axis("off")
            axes[i][j * 2 + 1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_loss_plot(lr_histories, output_dir, method_name, best_lr,
                   skip_epochs=0):
    """Save a publication-quality convergence plot showing all LR curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.tab10
    for i, lr in enumerate(sorted(lr_histories.keys())):
        history = [(e, l) for e, l in lr_histories[lr] if e > skip_epochs]
        if not history:
            continue
        epochs = [h[0] for h in history]
        losses = [h[1] for h in history]
        is_best = (lr == best_lr)
        lw = 2.5 if is_best else 1.0
        alpha = 1.0 if is_best else 0.4
        label = f"lr={lr:.0e}"
        if is_best:
            label += " (best)"
        ax.plot(epochs, losses, linewidth=lw, alpha=alpha, color=cmap(i),
                label=label)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(f"{method_name.upper()} Training Loss", fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=200, bbox_inches="tight")
    plt.close()

    save_data = {f"{lr:.0e}": history for lr, history in lr_histories.items()}
    save_data["best_lr"] = best_lr
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(save_data, f, indent=2)

    log.info("  Saved convergence plot -> %s", output_dir / "convergence.png")


def save_convergence_plot(history, output_dir):
    """Save OT solver convergence plot (gradient norm)."""
    from .ot_solver import GRAD_NORM_THRESHOLD

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot(history["step"], history["g_norm"], ".-", markersize=3, color="tab:green")
    ax.axhline(GRAD_NORM_THRESHOLD, color="red", linestyle="--",
               label=f"threshold={GRAD_NORM_THRESHOLD}")
    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Gradient Norm", fontsize=13)
    ax.set_title("AE-OT Convergence", fontsize=14, fontweight='bold')
    ax.set_yscale("log")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=200, bbox_inches="tight")
    plt.close()
