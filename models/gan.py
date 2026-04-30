import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerMLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, out_dim)
    )
  
  def forward(self, x):
    return self.layers(x)


class GAN(nn.Module):
  def __init__(self, data_shape):
    super().__init__()
    self.data_shape = data_shape

    in_dim = torch.prod(torch.tensor(data_shape)).item()
    self.generator = ThreeLayerMLP(in_dim, hidden_dim=128, out_dim=in_dim)
    self.discriminator = ThreeLayerMLP(in_dim, hidden_dim=128, out_dim=1)

  def generate(self, n):
    noise = torch.randn(n, *self.data_shape, device=next(self.parameters()).device)
    fake_data = self.generator(noise)
    return fake_data

  def discriminator_loss(self, x):
    # Use binary cross-entropy with logits (stable version) instead of
    #   explicit `nn.Sigmoid()` + `torch.log` to avoid NaN loss.
    # BCEWithLogits = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
    logits_real = self.discriminator(x)  # D(x) (logits)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))  # -log(D(x))

    N = x.shape[0]
    fake_data = self.generate(N)  # G(z)
    logits_fake = self.discriminator(fake_data)  # D(G(z)) (logits)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))  # -log(1-D(G(z)))

    d_loss = loss_real + loss_fake
    return d_loss

  def generator_loss(self, n):
    fake_data = self.generate(n)  # G(z)
    logits_fake = self.discriminator(fake_data)  # D(G(z)) (logits)
    # min log(1-D(G(Z)))
    g_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))  # -log(1-D(G(z)))
    g_loss = -g_loss
    return g_loss

  def sample(self, num_samples):
    with torch.no_grad():
      return self.generate(num_samples)
  
  def d_scores(self, data):
    with torch.no_grad():
      logits = self.discriminator(data)
      probs = torch.sigmoid(logits)
      return probs