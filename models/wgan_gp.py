import torch
import torch.nn as nn


class DepthToSpace(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    self.block_size = block_size
    self.block_size_sq = block_size * block_size

  def forward(self, input):
    output = input.permute(0, 2, 3, 1)
    (batch_size, d_height, d_width, d_depth) = output.size()
    s_depth = int(d_depth / self.block_size_sq)
    s_width = int(d_width * self.block_size)
    s_height = int(d_height * self.block_size)
    t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
    spl = t_1.split(self.block_size, 3)
    stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
    output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
    output = output.permute(0, 3, 1, 2)
    return output


class SpaceToDepth(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    self.block_size = block_size
    self.block_size_sq = block_size * block_size

  def forward(self, input):
    output = input.permute(0, 2, 3, 1)
    (batch_size, s_height, s_width, s_depth) = output.size()
    d_depth = s_depth * self.block_size_sq
    d_width = int(s_width / self.block_size)
    d_height = int(s_height / self.block_size)
    t_1 = output.split(self.block_size, 2)
    stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
    output = torch.stack(stack, 1)
    output = output.permute(0, 2, 1, 3)
    output = output.permute(0, 3, 1, 2)
    return output


# Spatial Upsampling with Nearest Neighbors
class UpsampleConv2d(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
    super().__init__()
    self.depth_to_space = DepthToSpace(block_size=2)
    self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

  def forward(self, x):
    x = torch.cat([x, x, x, x], dim=1)
    x = self.depth_to_space(x)
    x = self.conv2d(x)
    return x


# Spatial Downsampling with Spatial Mean Pooling
class DownsampleConv2d(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
    super().__init__()
    self.space_to_depth = SpaceToDepth(2)
    self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

  def forward(self, x):
    x = self.space_to_depth(x)
    x = sum(x.chunk(4, dim=1)) / 4.0
    x = self.conv2d(x)
    return x        


class ResnetBlockUp(nn.Module):
  def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
    super().__init__()
    self.residual_layers = nn.Sequential(
      nn.BatchNorm2d(in_dim),
      nn.ReLU(),
      nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
      nn.BatchNorm2d(n_filters),
      nn.ReLU(),
      UpsampleConv2d(n_filters, n_filters, kernel_size, padding=1)
    )
    self.shortcut_layer = UpsampleConv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

  def forward(self, x):
    residual = self.residual_layers(x)
    shortcut = self.shortcut_layer(x)
    return residual + shortcut


class ResnetBlockDown(nn.Module):
  def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
    super().__init__()
    self.residual_layers = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
      nn.ReLU(),
      DownsampleConv2d(n_filters, n_filters, kernel_size, padding=1)
    )
    self.shortcut_layer = DownsampleConv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

  def forward(self, x):
    residual = self.residual_layers(x)
    shortcut = self.shortcut_layer(x)
    return residual + shortcut


class ResBlock(nn.Module):
  def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
    super().__init__()
    self.residual_layers = nn.Sequential(
      nn.BatchNorm2d(in_dim),
      nn.ReLU(),
      nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
      nn.BatchNorm2d(n_filters),
      nn.ReLU(),
      nn.Conv2d(n_filters, n_filters, kernel_size, padding=1)
    )
    self.shortcut_layer = nn.Conv2d(n_filters, n_filters, kernel_size=(1, 1), padding=0)

  def forward(self, x):
    residual = self.residual_layers(x)
    shortcut = self.shortcut_layer(x)
    return residual + shortcut


class GlobalSumPooling(nn.Module):
  def forward(self, x):
    return torch.sum(x, dim=(2, 3))


class WGANGP(nn.Module):
  def __init__(self, n_filters, gradient_penalty_lambda):
    super().__init__()
    self.lambda_gp = gradient_penalty_lambda

    self.generator = nn.Sequential(
      nn.Linear(128, 4*4*256),
      nn.Unflatten(1, (256, 4, 4)),   # reshape output of linear layer
      ResnetBlockUp(in_dim=256, n_filters=n_filters),
      ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
      ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
      nn.BatchNorm2d(n_filters),
      nn.ReLU(),
      nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
      nn.Tanh()
    )
    self.discriminator = nn.Sequential(
        ResnetBlockDown(3, n_filters=n_filters),
        ResnetBlockDown(128, n_filters=n_filters),
        ResBlock(n_filters, n_filters=n_filters),
        ResBlock(n_filters, n_filters=n_filters),
        nn.ReLU(),
        GlobalSumPooling(),  # global sum pooling
        nn.Linear(128, 1)
    )

  def generate(self, n):
    noise = torch.randn(n, 128, device=next(self.parameters()).device)
    fake_data = self.generator(noise)
    return fake_data

  def discriminator_loss(self, x):
    logits_real = self.discriminator(x)  # D(x)
    loss_real = torch.mean(logits_real)

    N = x.shape[0]
    fake_data = self.generate(N)
    logits_fake = self.discriminator(fake_data)  # D(G(z))
    loss_fake = torch.mean(logits_fake)

    eps = torch.rand(N, 1, 1, 1, device=next(self.parameters()).device)
    x_interp = eps * x + (1 - eps) * fake_data
    logits_interp = self.discriminator(x_interp)  # D(x_hat)
    gradients = torch.autograd.grad(outputs=logits_interp, inputs=x_interp,
                                    grad_outputs=torch.ones_like(logits_interp), create_graph=True)[0]
    gradients = gradients.reshape(N, -1)  # (N, C, H, W) -> (N, C*H*W)
    grad_norm = torch.norm(gradients, dim=1)  # (N,)
    gradient_penalty = torch.mean((grad_norm - 1)**2)

    d_loss = loss_fake - loss_real + self.lambda_gp * gradient_penalty
    return d_loss

  def generator_loss(self, n):
    fake_data = self.generate(n)
    logits_fake = self.discriminator(fake_data)  # D(G(z))
    g_loss = -torch.mean(logits_fake)  # -D(G(z))
    return g_loss

  def sample(self, num_samples):
    with torch.no_grad():
      return self.generate(num_samples)