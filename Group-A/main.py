"""
Group A: 2D MNIST latent space visualization.

Visualizes 2D embeddings and mappings to [0,1]^2 for comparing
how well each method covers the target space.

Usage:
    cd ot-python/Group-A
    python main.py
"""

import logging
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets as tv_datasets

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist():
    train = tv_datasets.MNIST(root="./data", train=True, download=True)
    test = tv_datasets.MNIST(root="./data", train=False, download=True)
    X = np.concatenate([
        train.data.numpy().reshape(-1, 784),
        test.data.numpy().reshape(-1, 784),
    ]).astype(np.float32) / 255.0
    y = np.concatenate([train.targets.numpy(), test.targets.numpy()])
    log.info("Loaded MNIST: %d samples", len(X))
    return X, y


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    experiments = config["experiments"]
    output_dir = Path("output") / config["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_mnist()
    Z_umap = None
    Z_ae = None

    for name in experiments:
        log.info("\n%s", "=" * 60)
        log.info("  Experiment: %s", name)
        log.info("%s", "=" * 60)

        set_seed(SEED)
        mod = __import__(f"src.{name}", fromlist=["run"])

        if name == "umap":
            Z_umap = mod.run(X, y, output_dir)
        elif name in ("cnf", "ddpm", "fm", "nf", "otemd", "sdot", "otswd"):
            mod.run(Z_umap, y, output_dir)
        elif name == "ae":
            Z_ae = mod.run(X, y, output_dir)
        else:
            mod.run(X, y, output_dir)

    log.info("\n%s", "=" * 60)
    log.info("  Group A complete. Results in: %s/", output_dir)
    log.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
