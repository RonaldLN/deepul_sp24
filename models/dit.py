import torch
import torch.nn as nn
from .transformer import MultiHeadAttention, FeedForwardNetwork
from .vit import PatchEmbedding, PatchToImage
from .unet import TimestepEmbedding


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
  assert embed_dim % 2 == 0

  # use half of dimensions to encode grid_h
  emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
  emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

  emb = torch.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
  return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
  assert embed_dim % 2 == 0
  omega = torch.arange(embed_dim // 2, dtype=torch.float64)
  omega /= embed_dim / 2.
  omega = 1. / 10000**omega  # (D/2,)

  pos = pos.reshape(-1)  # (M,)
  out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

  emb_sin = torch.sin(out) # (M, D/2)
  emb_cos = torch.cos(out) # (M, D/2)

  emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
  return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
  grid_h = torch.arange(grid_size, dtype=torch.float32)
  grid_w = torch.arange(grid_size, dtype=torch.float32)
  grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
  grid = torch.stack(grid, axis=0)

  grid = grid.reshape([2, 1, grid_size, grid_size])
  pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
  return pos_embed


def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(MultiHeadAttention):
  def forward(self, x):
    return super().forward(x, x, x)


class DiTBlock(nn.Module):
  def __init__(self, hidden_size, num_heads):
    super().__init__()
    self.c_proj = nn.Sequential(
      nn.SiLU(),
      nn.Linear(hidden_size, 6 * hidden_size)
    )
    self.norm_attn = nn.LayerNorm(hidden_size, elementwise_affine=False)
    self.attn = Attention(hidden_size, num_heads)
    self.norm_ffn = nn.LayerNorm(hidden_size, elementwise_affine=False)
    self.ffn = FeedForwardNetwork(hidden_size, 4 * hidden_size)

  def forward(self, x, c):  # Given x (B x L x D), c (B x D)
    c = self.c_proj(c)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)

    h = self.norm_attn(x)
    h = modulate(h, shift_msa, scale_msa)
    x = x + gate_msa.unsqueeze(1) * self.attn(h)

    h = self.norm_ffn(x)
    h = modulate(h, shift_mlp, scale_mlp)
    x = x + gate_mlp.unsqueeze(1) * self.ffn(h)

    return x


class FinalLayer(nn.Module):
  def __init__(self, hidden_size, patch_size, out_channels):
    super().__init__()
    self.c_proj = nn.Sequential(
      nn.SiLU(),
      nn.Linear(hidden_size, 2 * hidden_size)
    )
    self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
    # PatchToImage already contains proj, remove it here

  def forward(self, x, c):  # Given x (B x L x D), c (B x D)
    c = self.c_proj(c)
    shift, scale = c.chunk(2, dim=1)
    x = self.norm(x)
    x = modulate(x, shift, scale)
    return x


class ClassifierFreeGuidanceDropout(nn.Module):
  def __init__(self, cfg_dropout_prob, num_classes):
    super().__init__()
    self.cfg_dropout_prob = cfg_dropout_prob
    self.num_classes = num_classes
    self.null_class_idx = num_classes

  def forward(self, y: torch.Tensor):
    p = torch.ones_like(y) * self.cfg_dropout_prob
    mask = torch.bernoulli(p)
    y = y.masked_fill(mask == 1, self.null_class_idx)
    return y


class DiT(nn.Module):
  def __init__(self, input_shape, patch_size, hidden_size, num_heads,
               num_layers, num_classes, cfg_dropout_prob):
    super().__init__()
    C, H, W = input_shape
    self.patch_embed = PatchEmbedding(H, patch_size, C, hidden_size)
    self.register_buffer("pos_embed", get_2d_sincos_pos_embed(hidden_size, H // patch_size).float())
    self.time_embed = TimestepEmbedding(hidden_size)
    self.cfg_dropout = ClassifierFreeGuidanceDropout(cfg_dropout_prob, num_classes)
    self.class_embed = nn.Embedding(num_classes + 1, hidden_size)
    self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])
    self.final_layer = FinalLayer(hidden_size, patch_size, C)
    self.unpatchify = PatchToImage(H, patch_size, C, hidden_size)

  def forward(self, x, y, t):  # Given x (B x C x H x W) - image, y (B) - class label, t (B) - diffusion timestep
    x = self.patch_embed(x) # B x C x H x W -> B x (H // P * W // P) x D, P is patch_size
    x = x + self.pos_embed # see get_2d_sincos_pos_embed

    t = self.time_embed(t) # Same as in UNet
    if self.training:
      y = self.cfg_dropout(y) # Randomly dropout to train unconditional image generation
    y = self.class_embed(y)
    c = t + y

    for block in self.blocks:
      x = block(x, c)

    x = self.final_layer(x, c)
    x = self.unpatchify(x) # B x (H // P * W // P) x (P * P * C) -> B x C x H x W
    return x