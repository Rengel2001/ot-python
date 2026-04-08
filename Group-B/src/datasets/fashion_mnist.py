"""Fashion-MNIST dataset: 70k images (train + test), 1x28x28, [0, 1]."""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import logging
from . import DatasetConfig

log = logging.getLogger(__name__)


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform)
    X_train = torch.stack([img for img, _ in train_ds])
    X_test = torch.stack([img for img, _ in test_ds])
    X = torch.cat([X_train, X_test], dim=0)  # (70000, 1, 28, 28)
    log.info("Dataset 'fashion_mnist': %d images, shape %s, range [%.1f, %.1f]",
             len(X), X.shape, X.min(), X.max())
    return X.numpy()


def get_config():
    return DatasetConfig(
        name="fashion_mnist",
        dim_c=1, img_h=28, img_w=28,
        n_expected=70000,
        bat_size_P=70000,
        num_gen=10000,
        load_fn=load_data,
    )
