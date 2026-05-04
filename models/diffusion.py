import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from .unet import *


class ContinuousDiffusion(nn.Module):
  def __init__(self, model, data_shape):
    super().__init__()
    self.data_shape = data_shape

    self.model = model

  def predict_eps(self, x, t):
    N = x.shape[0]
    if t.dim() == 0:  # sampling
      t = t.expand(N, 1)

    # Expand t to match the spatial dimensions H, W of x
    if x.dim() > 2:
      t = t.expand(-1, -1, *x.shape[2:])

    x_with_t = torch.cat((x, t), dim=1)
    eps_hat = self.model(x_with_t)
    return eps_hat

  def loss(self, x):
    N = x.shape[0]
    t = torch.rand(N, device=next(self.parameters()).device)
    t = t.view(-1, *(1,) * (x.dim() - 1))  # (N,) -> (N, 1, 1, ...) for broadcasting with x
    alpha_t = torch.cos(torch.pi/2 * t)
    sigma_t = torch.sin(torch.pi/2 * t)
    eps = torch.randn_like(x)
    x_t = alpha_t * x + sigma_t * eps

    eps_hat = self.predict_eps(x_t, t)
    loss = F.mse_loss(eps, eps_hat)
    return loss

  def prepare_coeffs(self, num_steps):
    coeffs = {}
    ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1, device=next(self.parameters()).device)
    coeffs["ts"] = ts
    alphas = torch.cos(torch.pi/2 * ts)
    sigmas = torch.sin(torch.pi/2 * ts)
    coeffs["alphas"] = alphas
    coeffs["sigmas"] = sigmas

    # compute eta_t
    alphas_sq = alphas ** 2
    alphas_sq_t, alphas_sq_tm1 = alphas_sq[:-1], alphas_sq[1:]
    sigmas_t, sigmas_tm1 = sigmas[:-1], sigmas[1:]
    etas = sigmas_tm1 / sigmas_t * torch.sqrt(1 - alphas_sq_t / alphas_sq_tm1)
    coeffs["etas"] = etas

    return coeffs

  def ddpm_update(self, x, eps_hat, i, coeffs):
    alpha_t = coeffs["alphas"][i]
    alpha_tm1 = coeffs["alphas"][i+1]
    sigma_t = coeffs["sigmas"][i]
    sigma_tm1 = coeffs["sigmas"][i+1]
    eta_t = coeffs["etas"][i]
    eps_t = torch.randn_like(x)
    var = torch.clamp(sigma_tm1**2 - eta_t**2, min=0.0)  # clip sigma_t-1^2 - eta_t^2
    x_tm1 = alpha_tm1 * ((x - sigma_t * eps_hat) / alpha_t) \
          + torch.sqrt(var) * eps_hat \
          + eta_t * eps_t
    return x_tm1

  def sample(self, num_samples, num_steps):
    x = torch.randn(num_samples, *self.data_shape, device=next(self.parameters()).device)
    with torch.no_grad():
      coeffs = self.prepare_coeffs(num_steps)
      for i in tqdm(range(num_steps), desc="Generating samples (steps)"):
        t = coeffs["ts"][i]
        eps_hat = self.predict_eps(x, t)
        x = self.ddpm_update(x, eps_hat, i, coeffs)
    return x


class ContinuousDiffusionWithMLP(ContinuousDiffusion):
  def __init__(self, in_dim, hidden_dim):
    model = nn.Sequential(
      nn.Linear(in_dim + 1, hidden_dim),  # +1 for conditioning on t
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, in_dim)
    )
    data_shape = (in_dim,)
    super().__init__(model, data_shape)


class ContinuousDiffusionWithUNet(ContinuousDiffusion):
  def __init__(self, in_channels, hidden_dims, blocks_per_dim, data_shape):
    unet = UNet(in_channels, hidden_dims, blocks_per_dim)
    super().__init__(unet, data_shape)

  def predict_eps(self, x, t):
    N = x.shape[0]
    # UNet needs a t of shape (N,)
    if t.dim() == 0:  # sampling
      t = t.expand(N)
    elif t.dim() > 1:  # training
      t = t.view(N)  # (N, 1, 1, ...) -> (N,)
    return self.model(x, t)

  def ddpm_update(self, x, eps_hat, i, coeffs):
    alpha_t = coeffs["alphas"][i]
    alpha_tm1 = coeffs["alphas"][i+1]
    sigma_t = coeffs["sigmas"][i]
    sigma_tm1 = coeffs["sigmas"][i+1]
    eta_t = coeffs["etas"][i]
    eps_t = torch.randn_like(x)
    var = torch.clamp(sigma_tm1**2 - eta_t**2, min=0.0)  # clip sigma_t-1^2 - eta_t^2
    x_hat = (x - sigma_t * eps_hat) / alpha_t
    x_hat = torch.clamp(x_hat, min=-1.0, max=1.0)  # clip x_hat to [-1, 1]
    x_tm1 = alpha_tm1 * x_hat \
          + torch.sqrt(var) * eps_hat \
          + eta_t * eps_t
    return x_tm1