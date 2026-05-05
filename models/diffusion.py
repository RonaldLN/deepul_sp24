import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from .unet import *
from .dit import *


class ContinuousDiffusion(nn.Module):
  def __init__(self, model, data_shape):
    super().__init__()
    self.data_shape = data_shape

    self.model = model

  def cfg_predict_eps(self, x, t, *args, cfg_scale):
    y = args[0]
    y_null = torch.full_like(y, self.null_class_idx)

    eps_hat_cond = self.model(x, *args, t)
    eps_hat_uncond = self.model(x, y_null, t)

    eps_hat = eps_hat_uncond + cfg_scale * (eps_hat_cond - eps_hat_uncond)
    return eps_hat

  def predict_eps(self, x, t, *args, cfg_scale=None):
    if cfg_scale is not None:
      return self.cfg_predict_eps(x, t, *args, cfg_scale=cfg_scale)

    eps_hat = self.model(x, *args, t)
    return eps_hat

  def loss(self, x, *args):
    N = x.shape[0]
    t = torch.rand(N, device=next(self.parameters()).device)
    t_broadcast = t.view(-1, *(1,) * (x.dim() - 1))  # (N,) -> (N, 1, 1, ...) for broadcasting with x
    alpha_t = torch.cos(torch.pi/2 * t_broadcast)
    sigma_t = torch.sin(torch.pi/2 * t_broadcast)
    eps = torch.randn_like(x)
    x_t = alpha_t * x + sigma_t * eps

    eps_hat = self.predict_eps(x_t, t, *args)
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
        t = t.expand(num_samples)  # (N,)
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

  def predict_eps(self, x, t):
    t = t.view(-1, 1)
    x_with_t = torch.cat((x, t), dim=1)
    eps_hat = self.model(x_with_t)
    return eps_hat


class ContinuousDiffusionWithUNet(ContinuousDiffusion):
  def __init__(self, in_channels, hidden_dims, blocks_per_dim, data_shape):
    unet = UNet(in_channels, hidden_dims, blocks_per_dim)
    super().__init__(unet, data_shape)

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


class ContinuousDiffusionWithTransformer(ContinuousDiffusion):
  def __init__(self, input_shape, patch_size, hidden_size, num_heads,
               num_layers, num_classes, cfg_dropout_prob):
    model = DiT(input_shape, patch_size, hidden_size, num_heads,
               num_layers, num_classes, cfg_dropout_prob)
    super().__init__(model, input_shape)
    self.null_class_idx = num_classes

  def ddpm_update(self, x, eps_hat, i, coeffs):
    alpha_t = coeffs["alphas"][i]
    alpha_tm1 = coeffs["alphas"][i+1]
    sigma_t = coeffs["sigmas"][i]
    sigma_tm1 = coeffs["sigmas"][i+1]
    eta_t = coeffs["etas"][i]
    eps_t = torch.randn_like(x)
    var = torch.clamp(sigma_tm1**2 - eta_t**2, min=0.0)  # clip sigma_t-1^2 - eta_t^2
    x_hat = (x - sigma_t * eps_hat) / alpha_t
    x_hat = torch.clamp(x_hat, min=-8.0, max=8.0)  # clip x_hat to [-8, 8]
    x_tm1 = alpha_tm1 * x_hat \
          + torch.sqrt(var) * eps_hat \
          + eta_t * eps_t
    return x_tm1

  def sample(self, num_samples, num_steps, class_idxs, cfg_scale=None):
    assert class_idxs.dim() == 1 and class_idxs.shape[0] == num_samples

    device = next(self.parameters()).device
    class_idxs = class_idxs.to(device)  # (N,)
    x = torch.randn(num_samples, *self.data_shape, device=device)
    with torch.no_grad():
      coeffs = self.prepare_coeffs(num_steps)
      for i in tqdm(range(num_steps), desc="Generating samples (steps)"):
        t = coeffs["ts"][i]
        t = t.expand(num_samples)  # (N,)
        eps_hat = self.predict_eps(x, t, class_idxs, cfg_scale=cfg_scale)
        x = self.ddpm_update(x, eps_hat, i, coeffs)
    return x