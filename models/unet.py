import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
  def __init__(self, dim, max_period=10000):
    super().__init__()
    self.dim = dim
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(half, dtype=torch.float32) / half)
    self.register_buffer("freqs", freqs)

  def forward(self, timesteps):
    args = timesteps[:, None].float() * self.freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if self.dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, temb_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.residual_layers1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.GroupNorm(num_groups=8, num_channels=out_channels),
      nn.SiLU()
    )
    self.temb_proj = nn.Linear(temb_channels, out_channels)
    self.residual_layers2 = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      nn.GroupNorm(num_groups=8, num_channels=out_channels),
      nn.SiLU()
    )
    if in_channels != out_channels:
      self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, x, temb):
    h = self.residual_layers1(x)
    temb = self.temb_proj(temb)
    h = h + temb[:, :, None, None]
    h = self.residual_layers2(h)
    if hasattr(self, "shortcut_conv"):
      x = self.shortcut_conv(x)
    x = x + h
    return x


class Downsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

  def forward(self, x):
    return self.conv(x)


class Upsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

  def forward(self, x):
    x = F.interpolate(x, scale_factor=2)
    x = self.conv(x)
    return x


class UNet(nn.Module):
  def __init__(self, in_channels, hidden_dims, blocks_per_dim):
    super().__init__()
    self.hidden_dims_len = len(hidden_dims)
    self.blocks_per_dim = blocks_per_dim
    temb_channels = hidden_dims[0] * 4

    self.time_embed = TimestepEmbedding(hidden_dims[0])
    self.time_mlp = nn.Sequential(
      nn.Linear(hidden_dims[0], temb_channels),
      nn.SiLU(),
      nn.Linear(temb_channels, temb_channels)
    )

    self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)

    prev_ch = hidden_dims[0]
    down_block_chans = [prev_ch]
    self.down_blocks = nn.ModuleList([])
    self.downsamples = nn.ModuleList([])
    for i, hidden_dim in enumerate(hidden_dims):
      for _ in range(blocks_per_dim):
        self.down_blocks.append(ResidualBlock(prev_ch, hidden_dim, temb_channels))
        prev_ch = hidden_dim
        down_block_chans.append(prev_ch)
      if i != len(hidden_dims) - 1:
        self.downsamples.append(Downsample(prev_ch))
        down_block_chans.append(prev_ch)

    self.middle_block1 = ResidualBlock(prev_ch, prev_ch, temb_channels)
    self.middle_block2 = ResidualBlock(prev_ch, prev_ch, temb_channels)

    self.up_blocks = nn.ModuleList([])
    self.upsamples = nn.ModuleList([])
    for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
      for j in range(blocks_per_dim + 1):
        dch = down_block_chans.pop()
        self.up_blocks.append(ResidualBlock(prev_ch + dch, hidden_dim, temb_channels))
        prev_ch = hidden_dim
        if i and j == blocks_per_dim:
          self.upsamples.append(Upsample(prev_ch))

    self.final_conv_block = nn.Sequential(
      nn.GroupNorm(num_groups=8, num_channels=prev_ch),
      nn.SiLU(),
      nn.Conv2d(prev_ch, in_channels, 3, padding=1)
    )

  def forward(self, x, t):
    emb = self.time_embed(t)
    emb = self.time_mlp(emb)

    h = self.init_conv(x)
    hs = [h]

    down_idx = 0
    for i in range(self.hidden_dims_len):
      for _ in range(self.blocks_per_dim):
        h = self.down_blocks[down_idx](h, emb)
        hs.append(h)
        down_idx += 1
      if i != self.hidden_dims_len - 1:
        h = self.downsamples[i](h)
        hs.append(h)

    h = self.middle_block1(h, emb)
    h = self.middle_block2(h, emb)

    up_idx = 0
    upsample_idx = 0
    for i in range(self.hidden_dims_len)[::-1]:
      for j in range(self.blocks_per_dim + 1):
        skip = hs.pop()
        h = torch.cat([h, skip], dim=1)
        h = self.up_blocks[up_idx](h, emb)
        up_idx += 1
        if i and j == self.blocks_per_dim:
          h = self.upsamples[upsample_idx](h)
          upsample_idx += 1

    out = self.final_conv_block(h)
    return out