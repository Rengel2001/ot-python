"""Dataset registry for Group B experiments."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DatasetConfig:
    name: str
    dim_c: int          # channels
    img_h: int          # height
    img_w: int          # width
    n_expected: int     # expected number of images
    bat_size_P: int     # OT source point batch size
    num_gen: int        # number of generated samples for FID
    load_fn: object     # callable returning (N, C, H, W) numpy array


def get_dataset(name: str) -> DatasetConfig:
    if name == "mnist":
        from .mnist import get_config
    elif name == "fashion_mnist":
        from .fashion_mnist import get_config
    elif name == "cifar10":
        from .cifar10 import get_config
    elif name == "celeba":
        from .celeba import get_config
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Available: mnist, fashion_mnist, cifar10, celeba")
    return get_config()


DATASETS = ["mnist", "fashion_mnist", "cifar10", "celeba"]
