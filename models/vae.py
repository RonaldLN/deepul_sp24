import torch
import torch.nn as nn


class TwoLayerFCNet(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, out_dim)
    )

  def forward(self, x):
    return self.layers(x)


class VAE(nn.Module):
  def __init__(self, in_dim, hidden_dim, latent_dim):
    super().__init__()
    self.in_dim = in_dim
    self.latent_dim = latent_dim

    self.encoder_mean = TwoLayerFCNet(in_dim, hidden_dim, latent_dim)
    self.encoder_var_log = TwoLayerFCNet(in_dim, hidden_dim, latent_dim)  # use log to keep positivity
    self.decoder_mean = TwoLayerFCNet(latent_dim, hidden_dim, in_dim)
    self.decoder_var_log = TwoLayerFCNet(latent_dim, hidden_dim, in_dim)

  def encode(self, x):
    z_mean = self.encoder_mean(x)
    z_var_log = self.encoder_var_log(x)
    z_var = torch.exp(z_var_log)  # (N, D)
    return z_mean, z_var

  def decode(self, z):
    x_mean = self.decoder_mean(z)
    x_var_log = self.decoder_var_log(z)
    x_var = torch.exp(x_var_log)
    return x_mean, x_var

  def reparameterize(self, mean, var):
    eps = torch.randn_like(var, device=next(self.parameters()).device)  # (N, D) sample independent noise for each (latent) dimension
    out = mean + eps * var**0.5
    return out

  def kl_divergence(self, z_mean, z_var):
    # KL = Eq[log(q) - log(p)]
    #    = -0.5 log(var) - 0.5 E[(z-mu)^2/var] + 0.5 E[z^2]
    #    = 0.5 * (-log(var) - 1 + mu^2 + var)
    z_var_log = torch.log(z_var)
    kl = 0.5 * (-z_var_log - 1 + z_mean**2 + z_var)  # (N, D)
    kl = torch.sum(kl, dim=1)  # (N,)
    kl = torch.mean(kl, dim=0)
    return kl

  def loss(self, x):
    z_mean, z_var = self.encode(x)
    z = self.reparameterize(z_mean, z_var)

    kl_loss = self.kl_divergence(z_mean, z_var)

    decoded_mean, decoded_var = self.decode(z)
    decoded_var_diag = torch.diag_embed(decoded_var)  # (N, D, D)
    decoded_mvn = torch.distributions.MultivariateNormal(decoded_mean, decoded_var_diag)
    log_probs = decoded_mvn.log_prob(x)  # (N,)
    recon_loss = torch.mean(-log_probs)

    neg_elbo = recon_loss + kl_loss

    return neg_elbo, recon_loss, kl_loss

  def sample(self, num_samples, with_noise=True):
    z = torch.randn((num_samples, self.latent_dim), device=next(self.parameters()).device)
    with torch.no_grad():
      decoded_mean, decoded_var = self.decode(z)
      if with_noise:
        x = self.reparameterize(decoded_mean, decoded_var)
      else:
        x = decoded_mean
    return x