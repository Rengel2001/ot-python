"""CelebA dataset: ~162k images (train split), 3x64x64, [0, 1].

Uses CenterCrop(108) + Resize(64) — standard DCGAN face crop.
"""

import logging
import numpy as np
import torchvision
import torchvision.transforms as transforms
from . import DatasetConfig

log = logging.getLogger(__name__)


def load_data():
    transform = transforms.Compose([
        transforms.CenterCrop(108),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CelebA(
        root="./data", split="train", download=True, transform=transform)
    log.info("Loading CelebA training split (%d images)...", len(dataset))
    X = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        X.append(img.numpy())
        if (i + 1) % 50000 == 0:
            log.info("  %d/%d loaded", i + 1, len(dataset))
    X = np.array(X, dtype=np.float32)
    log.info("Dataset 'celeba': %d images, shape %s, range [%.1f, %.1f]",
             len(X), X.shape, X.min(), X.max())
    return X


def get_config():
    return DatasetConfig(
        name="celeba",
        dim_c=3, img_h=64, img_w=64,
        n_expected=162770,
        bat_size_P=10000,
        num_gen=10000,
        load_fn=load_data,
    )
