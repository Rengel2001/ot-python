"""
Autoencoder using the InfoGAN architecture from Lucic et al. 2018.

Academic literature:
  1. Chen et al. 2016 published InfoGAN. The key contribution was the architecture
     (specific FC + deconv layer pattern), not just the adversarial training.
  2. Lucic et al. 2018 ("Are GANs Created Equal?") standardized on the InfoGAN
     architecture (compare_gan/architectures/infogan.py) as their network backbone
     for MNIST/Fashion-MNIST/CIFAR-10.
  3. Gu et al. 2020 (AE-OT paper) uses "the same architecture as Lucic et al."
     They took Lucic's InfoGAN network structure and trained it as an autoencoder
     with MSE reconstruction loss + L1 latent regularization.

Architecture (FC + 2 deconv layers, NOT all-conv):
  Encoder: input -> Conv(64,4,s2) -> Conv(128,4,s2) -> flatten -> FC(1024) -> FC(z_dim)
  Decoder: z -> FC(1024) -> FC(128*h4*w4) -> reshape -> DeConv(64,4,s2) -> DeConv(c,4,s2) -> Sigmoid

Parametrized by (dim_z, dim_c, img_h, img_w) to work for all datasets:
  MNIST:         dim_c=1, img_h=28, img_w=28 -> h4=7,  FC2=6272
  Fashion-MNIST: dim_c=1, img_h=28, img_w=28 -> h4=7,  FC2=6272
  CIFAR-10:      dim_c=3, img_h=32, img_w=32 -> h4=8,  FC2=8192
  CelebA:        dim_c=3, img_h=64, img_w=64 -> h4=16, FC2=32768
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, dim_z=100, dim_c=1, img_h=28, img_w=28):
        super().__init__()
        self.dim_z = dim_z
        self.dim_c = dim_c
        self.img_h = img_h
        self.img_w = img_w
        self.h4 = img_h // 4
        self.w4 = img_w // 4

        # Encoder: conv layers -> flatten -> FC layers
        self.enc_conv = nn.Sequential(
            nn.Conv2d(dim_c, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc_fc = nn.Sequential(
            nn.Linear(128 * self.h4 * self.w4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_z),
        )

        # Decoder: FC layers -> reshape -> deconv layers
        self.dec_fc = nn.Sequential(
            nn.Linear(dim_z, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 128 * self.h4 * self.w4),
            nn.BatchNorm1d(128 * self.h4 * self.w4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, dim_c, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        x = self.enc_conv(x)
        x = x.view(x.size(0), -1)
        z = self.enc_fc(x)
        return z.view(z.size(0), -1, 1, 1)  # (B, z_dim, 1, 1)

    def decoder(self, z):
        z = z.view(z.size(0), -1)
        x = self.dec_fc(z)
        x = x.view(x.size(0), 128, self.h4, self.w4)
        x = self.dec_conv(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
