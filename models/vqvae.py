import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEResidualBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.BatchNorm2d(dim),
      nn.ReLU(),
      nn.Conv2d(dim, dim, 3, 1, 1),
      nn.BatchNorm2d(dim),
      nn.ReLU(),
      nn.Conv2d(dim, dim, 1, 1, 0)
    )

  def forward(self, x):
    return self.layers(x) + x


class VQVAE(nn.Module):
  def __init__(self, k, beta=0.25):
    nn.Module.__init__(self)
    self.beta = beta

    self.encoder = nn.Sequential(
      nn.Conv2d(3, 256, 4, 2, 1),  # 16 x 16
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Conv2d(256, 256, 4, 2, 1),  # 8 x 8
      VQVAEResidualBlock(256),
      VQVAEResidualBlock(256)
    )
    self.decoder = nn.Sequential(
      VQVAEResidualBlock(256),
      VQVAEResidualBlock(256),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.ConvTranspose2d(256, 256, 4, 2, 1), # 16 x 16
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.ConvTranspose2d(256, 3, 4, 2, 1)  # 32 x 32
    )
    # Initialize codebook with uniform(-1/K, 1/K)
    self.codebook = nn.Parameter(torch.empty(k, 256).uniform_(-1/k, 1/k))

  def _quantize(self, x):
    z_e = self.encoder(x)
    N, z_c, z_h, z_w = z_e.shape
    # Permute channel dimension to the last first.
    # Direct reshape without permutation (`z_e.reshape(-1, 1, z_c)`) would break spatial structure
    z_e_flatten = z_e.permute(0, 2, 3, 1).reshape(-1, 1, z_c)  # (N, 256, H_z, W_z) -> (N, H_z, W_z, 256) -> (N * H_z * W_z, 1, 256)

    squared_distances = (z_e_flatten - self.codebook) ** 2  # (N', 1, 256) - (K, 256) -> (N', K, 256)
    squared_distances = torch.sum(squared_distances, dim=-1)  # (N', K)

    indices = torch.argmin(squared_distances, dim=1)  # (N',)
    indices = indices.view(N, z_h, z_w)  # (N, H_z, W_z)
    return indices, z_e

  def quantize(self, x):
    indices, _ = self._quantize(x)
    return indices

  def loss(self, x):
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

    return loss

  def decode(self, indices):
    z_q = self.codebook[indices]
    z_q = z_q.permute(0, 3, 1, 2)
    decoded_img = self.decoder(z_q)
    return decoded_img

  def reconstruct(self, x):
    x = x.to(next(self.parameters()).device)
    with torch.no_grad():
      indices, _ = self._quantize(x)
      recon_x = self.decode(indices)
      # Interleave original and reconstructed images along the batch dimension (shuffle-like)
      output = torch.stack((x, recon_x), dim=1)  # (N, 2, C, H, W)
      output = output.reshape(-1, *x.shape[1:])  # (N * 2, C, H, W)
    return output