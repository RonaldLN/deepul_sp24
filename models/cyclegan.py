import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
      nn.InstanceNorm2d(out_channels),
      nn.ReLU()
    )

  def forward(self, x):
    return self.layers(x)


class UpBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()
    self.layers = nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
      nn.InstanceNorm2d(out_channels),
      nn.ReLU()
    )

  def forward(self, x):
    return self.layers(x)


class CycleGANGenerator(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.init_conv = nn.Conv2d(in_channels, 32, 3, 1, 1)  # [32, 28, 28]
    self.down1 = ConvBlock(32, 64, 3, 2, 1)  # [64, 14, 14]
    self.down2 = ConvBlock(64, 64, 3, 2, 1)  # [64, 7, 7]
    self.mid = ConvBlock(64, 64, 3, 1, 1)  # [64, 7, 7]
    self.up1 = UpBlock(64, 64, 4, 2, 1)  # [64, 14, 14]
    self.up2 = UpBlock(64 + 64, 32, 4, 2, 1)  # [32, 28, 28]
    self.final_conv = nn.Conv2d(32, out_channels, 3, 1, 1)

  def forward(self, x):
    x = self.init_conv(x)
    x = self.down1(x)
    skip_shortcut = x
    x = self.down2(x)
    x = self.mid(x)
    x = self.up1(x)
    x = torch.cat((x, skip_shortcut), dim=1)
    x = self.up2(x)
    x = self.final_conv(x)
    return x


class CycleGANDiscriminator(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels, 128, 3, 1, 1),  # [128, 28, 28]
      nn.LeakyReLU(0.2),
      nn.Conv2d(128, 128, 3, 2, 1),  # [128, 14, 14]
      nn.LeakyReLU(0.2),
      nn.Conv2d(128, 128, 3, 2, 1),  # [128, 7, 7]
      nn.LeakyReLU(0.2),
      nn.Conv2d(128, 128, 3, 1, 1),  # [128, 7, 7]
      nn.LeakyReLU(0.2),
    )
    self.proj = nn.Linear(128, 1)

  def forward(self, x):
    x = self.layers(x)  # (N, 128, 7, 7)
    x = torch.sum(x, dim=(2, 3))  # (N, 128)
    x = self.proj(x)  # (N, 1)
    return x


class CycleGAN(nn.Module):
  def __init__(self, lambda_cyc=10):
    super().__init__()
    self.lambda_cyc = lambda_cyc

    self.generator1 = CycleGANGenerator(1, 3)  # G: X -> Y
    self.generator2 = CycleGANGenerator(3, 1)  # F: Y -> X
    self.discriminator1 = CycleGANDiscriminator(1)  # D_X
    self.discriminator2 = CycleGANDiscriminator(3)  # D_Y

  def discriminator_loss(self, x, translated_x, d_id):
    assert d_id == 1 or d_id == 2
    discriminator = self.discriminator1 if d_id == 1 else self.discriminator2

    translated_x = translated_x.detach()

    logits_real = discriminator(x)  # D(x) (logits)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))  # -log(D(x))
    logits_fake = discriminator(translated_x)  # D(x_hat) (logits)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))  # -log(1-D(x_hat))

    d_loss = loss_real + loss_fake
    return d_loss

  def cycle_loss(self, x, y, G_x, F_y):
    recon_x = self.generator2(G_x)  # F(G(x))
    recon_y = self.generator1(F_y)  # G(F(y))

    loss_x = F.l1_loss(x, recon_x)
    loss_y = F.l1_loss(y, recon_y)
    cycle_loss = loss_x + loss_y
    return cycle_loss

  def generator_loss(self, x, y, G_x, F_y):
    logits_G_x = self.discriminator2(G_x)  # D_Y(G(x))
    logits_F_y = self.discriminator1(F_y)  # D_X(F(y))
    g1_loss = F.binary_cross_entropy_with_logits(logits_G_x, torch.ones_like(logits_G_x))
    g2_loss = F.binary_cross_entropy_with_logits(logits_F_y, torch.ones_like(logits_F_y))

    cycle_loss = self.cycle_loss(x, y, G_x, F_y)

    g_loss = g1_loss + g2_loss + self.lambda_cyc * cycle_loss
    return g_loss

  def translate1(self, x):
    with torch.no_grad():
      return self.generator1(x)

  def translate2(self, x):
    with torch.no_grad():
      return self.generator2(x)