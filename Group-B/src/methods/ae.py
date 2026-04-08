"""
AE baseline: Generate by sampling from a Gaussian fitted to the latent codes.

This is the standard AE generative baseline. A multivariate Gaussian is fit to
the AE's latent codes (mean + covariance), then new codes are sampled from it.
Samples from low-density regions of latent space decode to poor images, which
is exactly the weakness that transport methods (OT, NF, FM, etc.) address.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)

NAME = "ae"


def generate(Z, latent_dim, device, output_dir, dataset_config):
    n_gen = dataset_config.num_gen
    mu = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False)
    gen_codes = np.random.multivariate_normal(mu, cov, size=n_gen).astype(np.float32)
    log.info("AE baseline: sampled %d codes from fitted Gaussian", n_gen)
    return gen_codes
