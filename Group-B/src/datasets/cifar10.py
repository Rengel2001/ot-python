"""CIFAR-10 dataset: 60k images (train + test), 3x32x32, [0, 1]."""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import logging
from . import DatasetConfig

log = logging.getLogger(__name__)


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)
    X_train = torch.stack([img for img, _ in train_ds])
    X_test = torch.stack([img for img, _ in test_ds])
    X = torch.cat([X_train, X_test], dim=0)  # (60000, 3, 32, 32)
    log.info("Dataset 'cifar10': %d images, shape %s, range [%.1f, %.1f]",
             len(X), X.shape, X.min(), X.max())
    return X.numpy()


def get_config():
    return DatasetConfig(
        name="cifar10",
        dim_c=3, img_h=32, img_w=32,
        n_expected=60000,
        bat_size_P=60000,
        num_gen=10000,
        load_fn=load_data,
    )
