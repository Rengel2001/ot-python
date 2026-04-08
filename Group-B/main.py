"""
Group B: 100-dim generation comparison across 4 datasets.

Trains AEs first (skipping if checkpoint exists), then runs all methods,
computes FID, and prints a comparison table.

Usage:
    cd ot-python/Group-B
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import json
import numpy as np
import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import get_dataset, DATASETS
from methods import get_method, METHODS
from core.training import (ae_lr_sweep, load_ae_checkpoint,
                            set_seed, decode_latent_codes, save_sample_images, SEED)
from core.fid_computation import compute_fid_and_cleanup

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger()


def run_method(dataset_name, method_name, device, X_images):
    cfg = get_dataset(dataset_name)
    output_dir = Path("output/results") / cfg.name / method_name

    if (output_dir / "result.json").exists():
        log.info("  Skipping %s/%s — result exists", dataset_name, method_name)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "logs.txt", mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(fh)

    set_seed(SEED)
    method_mod = get_method(method_name)
    N = len(X_images)

    log.info("\n" + "=" * 60)
    log.info("  Method: %s | Dataset: %s | N=%d", method_name, cfg.name, N)
    log.info("=" * 60)

    if method_name != "vae":
        ae_model, Z, latent_dim = load_ae_checkpoint(cfg.name, device)
        log.info("  Loaded AE: latent_dim=%d, Z=%s", latent_dim, Z.shape)
    else:
        latent_dim = 100
        Z = None

    if method_name == "vae":
        gen_images = method_mod.generate(
            Z, latent_dim, device, output_dir, cfg, X_images=X_images)
    else:
        gen_codes = method_mod.generate(Z, latent_dim, device, output_dir, cfg)
        gen_images = decode_latent_codes(
            ae_model, gen_codes[:cfg.num_gen], latent_dim, device)

    n_eval = min(len(gen_images), cfg.num_gen)
    n_real = min(cfg.num_gen, N)
    real_idx = np.random.choice(N, n_real, replace=False)
    final_fid = compute_fid_and_cleanup(
        X_images[real_idx], gen_images[:n_eval], device, output_dir)

    if method_name == "vae":
        title = f"VAE ({cfg.name})"
    elif method_name == "aeot":
        title = f"AE-OT ({cfg.name})"
    else:
        title = f"AE-{method_name.upper()} ({cfg.name})"
    save_sample_images(gen_images[:64], output_dir,
                       title=title, n_channels=cfg.dim_c)
    result = {"dataset": cfg.name, "method": method_name, "fid": final_fid}

    result["num_gen"] = cfg.num_gen
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    log.info("\n  FID: %.2f", final_fid)
    log.removeHandler(fh)
    fh.close()


def print_results():
    table = {}
    for dataset in DATASETS:
        table[dataset] = {}
        for method in METHODS:
            result_file = Path("output/results") / dataset / method / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    table[dataset][method] = json.load(f).get("fid")
            else:
                table[dataset][method] = None

    header = f"{'Dataset':<16}" + "".join(f"{'AE-'+m.upper():>10}" for m in METHODS)
    log.info("\n" + "=" * len(header))
    log.info("  FID Comparison (Group B -- 100-dim latent space)")
    log.info("=" * len(header))
    log.info(header)
    log.info("-" * len(header))
    for dataset in DATASETS:
        row = f"{dataset:<16}"
        for method in METHODS:
            fid = table[dataset].get(method)
            row += f"{fid:>10.2f}" if fid is not None else f"{'---':>10}"
        log.info(row)
    log.info("-" * len(header))

    total = len(DATASETS) * len(METHODS)
    done = sum(1 for d in DATASETS for m in METHODS
               if table.get(d, {}).get(m) is not None)
    log.info("Progress: %d/%d experiments complete", done, total)


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    datasets = config["datasets"]
    methods = config["methods"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Group B: %d datasets x %d methods | Device: %s",
             len(datasets), len(methods), device)

    log.info("\n" + "#" * 60)
    log.info("  PHASE 1: Train Autoencoders")
    log.info("#" * 60)
    failed_datasets = set()
    for dataset in datasets:
        try:
            ae_lr_sweep(get_dataset(dataset), device)
        except Exception as e:
            log.error("  FAILED %s: %s — skipping", dataset, e)
            failed_datasets.add(dataset)

    log.info("\n" + "#" * 60)
    log.info("  PHASE 2: Run Generation Methods")
    log.info("#" * 60)
    for dataset in datasets:
        if dataset in failed_datasets:
            log.info("  Skipping %s — AE training failed", dataset)
            continue
        cfg = get_dataset(dataset)
        ckpt = Path("output/models") / dataset / "ae_checkpoint.pt"
        try:
            X_images = cfg.load_fn()
        except Exception as e:
            log.error("  FAILED to load %s: %s — skipping", dataset, e)
            continue
        log.info("Loaded %s: %d images", dataset, len(X_images))

        for method in methods:
            if method != "vae" and not ckpt.exists():
                log.info("  Skipping %s/%s — no AE checkpoint", dataset, method)
                continue
            run_method(dataset, method, device, X_images)

        del X_images

    log.info("\n" + "#" * 60)
    log.info("  PHASE 3: Results")
    log.info("#" * 60)
    print_results()


if __name__ == "__main__":
    main()
