import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from deepul.hw3_utils.lpips import LPIPS
from .vqvae import *
from .wgan_gp import *
from .vit import *


class VQGANVQVAE(VQVAE):
  def loss(self, x):  # override to return additional intermediate variables
    indices, z_e = self._quantize(x)
    z_q = self.codebook[indices]  # (N, H_z, W_z) -> (N, H_z, W_z, C)
    z_q = z_q.permute(0, 3, 1, 2)  # (N, C, H_z, W_z)

    # straight-through estimator (given)
    z_q_st = (z_q - z_e).detach() + z_e
    recon_x = self.decoder(z_q_st)

    recon_loss = F.mse_loss(recon_x, x)
    vq_loss = F.mse_loss(z_q, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q.detach())

    loss = recon_loss + vq_loss + self.beta * commitment_loss

    return loss, recon_loss, recon_x


class VQGAN(nn.Module):
  def __init__(self, k, n_filters):
    super().__init__()

    self.vqvae = VQGANVQVAE(k)
    self.discriminator = nn.Sequential(
      ResnetBlockDown(3, n_filters=n_filters),
      ResnetBlockDown(128, n_filters=n_filters),
      ResBlock(n_filters, n_filters=n_filters),
      ResBlock(n_filters, n_filters=n_filters),
      nn.ReLU(),
      GlobalSumPooling(),  # global sum pooling
      nn.Linear(128, 1)
    )

    self._l_pips = LPIPS()

  def discriminate(self, x):
    # split to 8x8 patches
    N, C, H, W = x.shape
    x = x.view(N, C, H//8, 8, W//8, 8)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (N, H/8, W/8, C, 8, 8)
    x = x.reshape(-1, C, 8, 8)  # (N*H/8*W/8, C, 8, 8)

    logits = self.discriminator(x)  # (N*H/8*W/8, 1)
    logits = logits.view(N, H//8, W//8)
    return logits

  def discriminator_loss(self, x, recon_x):
    recon_x = recon_x.detach()  # Avoid second backward pass error

    logits_real = self.discriminate(x)  # D(x) (logits)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))  # -log(D(x))
    logits_fake = self.discriminate(recon_x)  # D(x_hat) (logits)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))  # -log(1-D(x_hat))

    d_loss = loss_real + loss_fake
    return d_loss

  def l_pips(self, x, recon_x):
    return self._l_pips(x, recon_x).mean()
  
  def vqvae_loss(self, x, recon_x, vq_loss, l2_loss):
    logits_fake = self.discriminate(recon_x)  # D(x_hat) (logits)
    # min log(1-D(x_hat)) => max log(D(x_hat)) => min -log(D(x_hat))
    gan_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))  # -log(D(x_hat))
    perceptual_loss = self.l_pips(x, recon_x)
    vqvae_loss = vq_loss + 0.1 * gan_loss + 0.5 * perceptual_loss + l2_loss
    return vqvae_loss, perceptual_loss, l2_loss

  def reconstruct(self, x):
    return self.vqvae.reconstruct(x)


class ViTVQGANVQVAE(VQGANVQVAE):
  def __init__(self, k, beta=0.25):
    nn.Module.__init__(self)
    self.beta = beta

    self.encoder = ViTVQGANEncoder(img_size=32, patch_size=4, in_channels=3, embed_dim=256,
                                   num_layers=4, num_heads=8, dim_feedforward=2*256, max_length=8*8, dropout=0.1)
    self.decoder = ViTVQGANDecoder(img_size=32, patch_size=4, in_channels=3, embed_dim=256,
                                   num_layers=4, num_heads=8, dim_feedforward=2*256, max_length=8*8, dropout=0.1)
    # # Initialize codebook with uniform(-1/K, 1/K)
    # self.codebook = nn.Parameter(torch.empty(k, 256).uniform_(-1/k, 1/k))
    self.codebook = nn.Parameter(torch.randn(k, 256))

  def _quantize(self, x):
    z_e = self.encoder(x)  # (N, H/P * W/P, D)
    N, _, D = z_e.shape
    z_e_flatten = z_e.view(-1, 1, D)  # (N * H/P * W/P, 1, 256)

    squared_distances = (z_e_flatten - self.codebook) ** 2  # (N', 1, 256) - (K, 256) -> (N', K, 256)
    squared_distances = torch.sum(squared_distances, dim=-1)  # (N', K)

    indices = torch.argmin(squared_distances, dim=1)  # (N',)
    indices = indices.view(N, -1)  # (N, H/P * W/P)
    return indices, z_e

  def loss(self, x):
    indices, z_e = self._quantize(x)
    z_q = self.codebook[indices]  # (N, H/P * W/P) -> (N, H/P * W/P, D)

    # straight-through estimator (given)
    z_q_st = (z_q - z_e).detach() + z_e
    recon_x = self.decoder(z_q_st)

    recon_loss = F.mse_loss(recon_x, x)
    vq_loss = F.mse_loss(z_q, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q.detach())

    loss = vq_loss + self.beta * commitment_loss

    return loss, recon_loss, recon_x

  def decode(self, indices):
    z_q = self.codebook[indices]  # (N, H/P * W/P) -> (N, H/P * W/P, D)
    decoded_img = self.decoder(z_q)
    return decoded_img


class ViTVQGAN(VQGAN):
  def __init__(self, k, n_filters):
    nn.Module.__init__(self)

    self.vqvae = ViTVQGANVQVAE(k)
    self.discriminator = nn.Sequential(
      ResnetBlockDown(3, n_filters=n_filters),
      ResnetBlockDown(128, n_filters=n_filters),
      ResBlock(n_filters, n_filters=n_filters),
      ResBlock(n_filters, n_filters=n_filters),
      nn.LeakyReLU(),
      GlobalSumPooling(),  # global sum pooling
      nn.Linear(128, 1)
    )
    # replace ReLU with LeakyReLU
    def replace_relu_with_leakyrelu(module):
      for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
          setattr(module, name, nn.LeakyReLU(0.2))
        else:
          replace_relu_with_leakyrelu(child)
    replace_relu_with_leakyrelu(self.discriminator)
    # apply spectral norm to weight matrices
    for m in self.discriminator.modules():
      if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        spectral_norm(m)

    self._l_pips = LPIPS()

  def discriminate(self, x):
    return self.discriminator(x)  # (N, C, H, W) -> (N, 1)
  
  def vqvae_loss(self, x, recon_x, vq_loss, l2_loss):
    logits_fake = self.discriminate(recon_x)  # D(x_hat) (logits)
    # min log(1-D(x_hat)) => max log(D(x_hat)) => min -log(D(x_hat))
    gan_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))  # -log(D(x_hat))
    perceptual_loss = self.l_pips(x, recon_x)
    l1_loss = F.l1_loss(x, recon_x)
    vqvae_loss = vq_loss + 0.1 * gan_loss + 0.5 * perceptual_loss + l2_loss + 0.1 * l1_loss
    return vqvae_loss, perceptual_loss, l2_loss