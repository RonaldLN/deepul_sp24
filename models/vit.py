import torch
import torch.nn as nn
from .transformer import *


# copy from cs231n
class PatchEmbedding(nn.Module):
  def __init__(self, img_size, patch_size, in_channels=3, embed_dim=256):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.embed_dim = embed_dim

    assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

    self.num_patches = (img_size // patch_size) ** 2  # H/P * W/P
    self.patch_dim = patch_size ** 2 * in_channels  # P * P * C

    self.proj = nn.Linear(self.patch_dim, embed_dim)

  def forward(self, x):
    N, C, H, W = x.shape
    assert H == self.img_size and W == self.img_size, \
      f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"

    P = self.patch_size
    x = x.reshape(N, C, H//P, P, W//P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (N, H/P, W/P, C, P, P)
    x = x.reshape(N, self.num_patches, self.patch_dim)  # (N, H/P * W/P, C*P*P)
    x = self.proj(x)  # (N, H/P * W/P, D)

    return x


class PatchToImage(nn.Module):
  def __init__(self, img_size, patch_size, in_channels=3, embed_dim=256):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.in_channels = in_channels
    self.embed_dim = embed_dim

    self.num_patches = (img_size // patch_size) ** 2  # H/P * W/P
    self.patch_dim = patch_size ** 2 * in_channels  # P * P * C

    self.proj = nn.Linear(embed_dim, self.patch_dim)

  def forward(self, x):
    C, H, W = self.in_channels, self.img_size, self.img_size
    P = self.patch_size
    x = self.proj(x)  # (N, H/P * W/P, C*P*P)
    x = x.reshape(-1, H//P, W//P, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (N, C, H/P, P, W/P, P)
    x = x.reshape(-1, C, H, W)
    return x


class ViTEncoderLayer(CausalTransformerDecoderLayer):
  pass


class ViTEncoder(CausalTransformerDecoder):
  pass


# largely copy from cs231n
class ViTVQGANEncoder(nn.Module):
  def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256,
               num_layers=4, num_heads=8, dim_feedforward=2*256, max_length=8*8, dropout=0.1):
    super().__init__()
    self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    self.positional_encoding = nn.Parameter(torch.randn(max_length, embed_dim))

    encoder_layer = ViTEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
    self.transformer = ViTEncoder(encoder_layer, num_layers)

  def forward(self, x):
    x = self.patch_embed(x)  # (N, H/P * W/P, D)
    num_patches = x.shape[1]
    x = x + self.positional_encoding[:num_patches]
    x = self.transformer(x)
    return x


class ViTVQGANDecoder(nn.Module):
  def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256,
               num_layers=4, num_heads=8, dim_feedforward=2*256, max_length=8*8, dropout=0.1):
    super().__init__()
    self.positional_encoding = nn.Parameter(torch.randn(max_length, embed_dim))

    encoder_layer = ViTEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
    self.transformer = ViTEncoder(encoder_layer, num_layers)

    self.patch_to_image = PatchToImage(img_size, patch_size, in_channels, embed_dim)

  def forward(self, x):
    num_patches = x.shape[1]
    x = x + self.positional_encoding[:num_patches]
    x = self.transformer(x)  # (N, H/P * W/P, D)
    x = self.patch_to_image(x)  # (N, C, H, W)
    return x