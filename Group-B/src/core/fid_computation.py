"""
FID computation using pytorch-fid (TF-ported InceptionV3 weights).

File-based: saves PNGs -> pytorch_fid.fid_score.calculate_fid_given_paths().
Grayscale images (MNIST, Fashion-MNIST) are converted to RGB via PIL .convert('RGB').

References:
  - Heusel et al., "GANs Trained by a Two Time-Scale Update Rule" (NeurIPS 2017)
  - pytorch-fid: https://github.com/mseitzer/pytorch-fid
  - Lucic et al., "Are GANs Created Equal?" (NeurIPS 2018)
"""

import shutil
import numpy as np
from pathlib import Path


def save_images_for_fid(images, output_dir):
    from PIL import Image
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        img = images[i]
        if img.shape[0] == 1:
            img_np = (img[0] * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L').convert('RGB')
        else:
            img_np = (np.transpose(img, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        pil_img.save(output_dir / f"{i:06d}.png")


def compute_fid(real_dir, gen_dir, device, batch_size=256):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return calculate_fid_given_paths(
        [str(real_dir), str(gen_dir)],
        batch_size=batch_size, device=device, dims=2048)


def compute_fid_and_cleanup(real_images, gen_images, device, output_dir):
    output_dir = Path(output_dir)
    real_dir = output_dir / "fid_real"
    gen_dir = output_dir / "fid_gen"

    save_images_for_fid(real_images, real_dir)
    save_images_for_fid(gen_images, gen_dir)
    fid = compute_fid(real_dir, gen_dir, device)

    shutil.rmtree(real_dir)
    shutil.rmtree(gen_dir)
    return fid


def compute_recon_fid(model, X_images, device, output_dir, n_samples=10000, batch_size=512):
    import torch

    model.eval()
    n_recon = min(n_samples, len(X_images))
    recon_idx = np.random.choice(len(X_images), n_recon, replace=False)

    recon_images = []
    with torch.no_grad():
        for i in range(0, n_recon, batch_size):
            batch = torch.tensor(
                X_images[recon_idx[i:i+batch_size]], dtype=torch.float32, device=device)
            z = model.encoder(batch)
            recon = model.decoder(z)
            recon_images.append(recon.cpu().numpy())
    recon_images = np.concatenate(recon_images)

    return compute_fid_and_cleanup(
        X_images[recon_idx], recon_images, device, output_dir)
