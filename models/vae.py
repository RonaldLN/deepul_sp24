import torch
import torch.nn as nn
import torch.nn.functional as F


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

    self.encoder = TwoLayerFCNet(in_dim, hidden_dim, 2 * latent_dim)  # 2*latent_dim includes mean and var
    self.decoder_mean = TwoLayerFCNet(latent_dim, hidden_dim, in_dim)
    self.decoder_var_log = TwoLayerFCNet(latent_dim, hidden_dim, in_dim)

  def encode(self, x):
    z_mean_var = self.encoder(x)
    z_mean, z_var_log = torch.split(z_mean_var, (self.latent_dim, self.latent_dim), dim=1)  # use log to keep positivity
    z_var = torch.exp(z_var_log)  # (N, D)
    return z_mean, z_var

  def decode(self, z):
    x_mean = self.decoder_mean(z)
    x_var_log = self.decoder_var_log(z)
    x_var = torch.exp(x_var_log)
    return x_mean, x_var

  def reparameterize(self, mean, var):
    eps = torch.randn_like(var)  # (N, D) sample independent noise for each (latent) dimension
    out = mean + eps * var**0.5
    return out

  def reparameterize_x(self, mean, var):
    return self.reparameterize(mean, var)

  def kl_divergence(self, z_mean, z_var):
    # KL = Eq[log(q) - log(p)]
    #    = -0.5 log(var) - 0.5 E[(z-mu)^2/var] + 0.5 E[z^2]
    #    = 0.5 * (-log(var) - 1 + mu^2 + var)
    z_var_log = torch.log(z_var)
    kl = 0.5 * (-z_var_log - 1 + z_mean**2 + z_var)  # (N, D)
    kl = torch.sum(kl, dim=1)  # (N,)
    kl = torch.mean(kl, dim=0)
    return kl

  def compute_recon_loss(self, x, decoded_mean, decoded_var):
    decoded_var_diag = torch.diag_embed(decoded_var)  # (N, D, D)
    decoded_mvn = torch.distributions.MultivariateNormal(decoded_mean, decoded_var_diag)
    log_probs = decoded_mvn.log_prob(x)  # (N,)
    recon_loss = torch.mean(-log_probs)
    return recon_loss

  def loss(self, x):
    z_mean, z_var = self.encode(x)
    z = self.reparameterize(z_mean, z_var)

    kl_loss = self.kl_divergence(z_mean, z_var)

    decoded_mean, decoded_var = self.decode(z)
    recon_loss = self.compute_recon_loss(x, decoded_mean, decoded_var)

    neg_elbo = recon_loss + kl_loss

    return neg_elbo, recon_loss, kl_loss

  def sample(self, num_samples, with_noise=True):
    z = torch.randn((num_samples, self.latent_dim), device=next(self.parameters()).device)
    with torch.no_grad():
      decoded_mean, decoded_var = self.decode(z)
      if with_noise:
        x = self.reparameterize_x(decoded_mean, decoded_var)
      else:
        zero_var = torch.zeros_like(decoded_mean)
        x = self.reparameterize_x(decoded_mean, zero_var)
    return x

  def reconstruct(self, x):
    x = x.to(next(self.parameters()).device)
    with torch.no_grad():
      z_mean, z_var = self.encode(x)
      z = self.reparameterize(z_mean, z_var)
      decoded_mean, decoded_var = self.decode(z)
      recon_x = self.reparameterize_x(decoded_mean, decoded_var)  # (N, D)
      recon_x = recon_x.view(x.shape)
      output = torch.cat((x, recon_x))
    return output

  def interpolate(self, x, n_interp):
    N = x.shape[0]
    x = x.to(next(self.parameters()).device)
    with torch.no_grad():
      x_start, x_end = torch.split(x, N//2)

      z_start_mean, z_start_var = self.encode(x_start)
      z_end_mean, z_end_var = self.encode(x_end)
      z_start = self.reparameterize(z_start_mean, z_start_var)
      z_end = self.reparameterize(z_end_mean, z_end_var)

      z_diff = z_end - z_start
      interp_ratios = torch.arange(n_interp, device=next(self.parameters()).device) / n_interp
      interp_ratios = interp_ratios.view(-1, *[1] * x.dim())  # (n_interp, 1, ..., 1) to broadcast with z_start and z_end
      z_interp = z_start + z_diff * interp_ratios  # (N, D) + (N, D) * (n_interp, 1, 1) -> (n_interp, N, D)
      z_interp = z_interp.view(n_interp * N//2, *z_start.shape[1:])

      decoded_mean, decoded_var = self.decode(z_interp)
      recon_x = self.reparameterize_x(decoded_mean, decoded_var)
      recon_x = recon_x.view(n_interp * N//2, *x.shape[1:])
    return recon_x


class ConvVAE(VAE):
  def __init__(self, image_c, image_h, image_w, latent_dim):
    nn.Module.__init__(self)
    self.image_c = image_c
    self.image_h = image_h
    self.image_w = image_w
    self.latent_dim = latent_dim

    self.encoder = nn.Sequential(
      nn.Conv2d(3, 32, 3, 1, 1),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 2, 1),  # 16 x 16
      nn.ReLU(),
      nn.Conv2d(64, 128, 3, 2, 1),  # 8 x 8
      nn.ReLU(),
      nn.Conv2d(128, 256, 3, 2, 1),  # 4 x 4
      nn.ReLU(),
      nn.Flatten(),  # 16
      nn.Linear(4 * 4 * 256, 2 * latent_dim)  # 2*latent_dim includes mean and var
    )
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, 4 * 4 * 128),
      nn.ReLU(),
      nn.Unflatten(1, (128, 4, 4)),
      nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 x 8
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 x 16
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 x 32
      nn.ReLU(),
      nn.Conv2d(32, 3, 3, 1, 1)
    )

  def decode(self, z):
    N = z.shape[0]
    x_mean = self.decoder(z)  # (N, C, H, W)
    x_var = torch.ones_like(x_mean)  # (N, C, H, W)
    return x_mean, x_var

  def reparameterize_x(self, mean, var):
    out = self.reparameterize(mean, var)
    out = torch.sigmoid(out)  # ensure the output is between 0 and 1
    return out

  def compute_recon_loss(self, x, decoded_mean, decoded_var):
    N = x.shape[0]
    recon_x = self.reparameterize_x(decoded_mean, decoded_var)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / N  # only average over the batch dimension
    return recon_loss


class HierVAEEncoderLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(3 + 12, 32, 3, padding=1), # [32, 32, 32]
      nn.LayerNorm([32, 32, 32]),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, stride=2, padding=1), # [64, 16, 16]
      nn.LayerNorm([64, 16, 16]),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 8, 8]
      nn.LayerNorm([64, 8, 8]),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 4, 4]
      nn.LayerNorm([64, 4, 4]),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 2, 2]
      nn.LayerNorm([64, 2, 2]),
      nn.ReLU(),
      nn.Conv2d(64, 12*2, 3, padding=1), # [12*2, 2, 2]
    )

  def forward(self, x):
    return self.layers(x)


class HierVAE(VAE):
  # Applying sigmoid after decoder output may keep KL loss staying near 0 during training.
  # A KL loss near 0 indicates the model is not learning meaningful features from data, 
  #   but merely fitting to a standard normal distribution.
  # The expected behavior is that KL loss first decreases, then slowly rises and stabilizes.
  # Using sigmoid causes gradients to vanish when input absolute values are large, 
  #   preventing gradients from propagating back through the decoder to the latent variable z.
  # Use the kl_weight coefficient to control the impact of the KL loss.
  def __init__(self, kl_weight=0.5):
    nn.Module.__init__(self)
    self.kl_weight = kl_weight

    self.encoder1 = HierVAEEncoderLayer()
    self.encoder2 = HierVAEEncoderLayer()
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(12, 64, 3, padding=1), # [64, 2, 2]
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # [64, 4, 4]
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # [64, 8, 8]
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # [64, 16, 16]
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # [32, 32, 32]
      nn.ReLU(),
      nn.Conv2d(32, 3, 3, padding=1), # [3, 32, 32]
    )
    # p(z2|z1) learn only the mean
    self.prior = nn.Sequential(
      nn.Conv2d(12, 64, 3, padding=1), # [64, 2, 2]
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, padding=1), # [64, 2, 2]
      nn.ReLU(),
      nn.Conv2d(64, 12, 3, padding=1) # [12, 2, 2]
    )

  def encode_helper(self, x):
    N, _, H, W = x.shape

    # x -> z1
    z0 = torch.zeros((N, 12, H, W), device=next(self.parameters()).device)
    z0_and_x = torch.cat((z0, x), dim=1)
    z1_mean_var = self.encoder1(z0_and_x)
    z1_mean, z1_var_log = torch.split(z1_mean_var, (12, 12), dim=1)
    z1_var = torch.exp(z1_var_log)
    z1 = self.reparameterize(z1_mean, z1_var)

    # z1, x -> z2
    z2_mean_prior = self.prior(z1)  # prior z2 mean given z1
    z2_var_prior = torch.ones_like(z2_mean_prior)
    # upscale z1 with nearest-neighbor projection
    z1_upscaled = F.interpolate(z1, size=(H, W), mode="nearest")
    z1_and_x = torch.cat((z1_upscaled, x), dim=1)
    z2_residual_mean_var = self.encoder2(z1_and_x)
    z2_residual_mean, z2_residual_var_log = torch.split(z2_residual_mean_var, (12, 12), dim=1)
    z2_residual_var = torch.exp(z2_residual_var_log)

    z2_mean = z2_mean_prior + z2_residual_mean
    z2_var = z2_var_prior * z2_residual_var

    # terms for kl_z2
    z2_residual_logstd = z2_residual_var_log / 2
    z2_residual_mu = z2_residual_mean

    return z1_mean, z1_var, z1_var_log, z2_mean, z2_var, z2_residual_logstd, z2_residual_mu

  def encode(self, x):
    _, _, _, z2_mean, z2_var, _, _ = self.encode_helper(x)
    return z2_mean, z2_var

  def decode(self, z):
    x = self.decoder(z)
    x_var = torch.zeros_like(x)  # set var=0 -> x = x_mean
    return x, x_var

  def loss(self, x):
    N = x.shape[0]

    z1_mean, z1_var, z1_var_log, z2_mean, z2_var, z2_residual_logstd, z2_residual_mu = self.encode_helper(x)
    z2 = self.reparameterize(z2_mean, z2_var)

    kl_z1 = 0.5 * (-z1_var_log - 1 + z1_mean**2 + z1_var)
    # formula for kl_z2 is given
    kl_z2 = -z2_residual_logstd - 0.5 + (torch.exp(2 * z2_residual_logstd) + z2_residual_mu ** 2) * 0.5
    kl = kl_z1 + kl_z2
    kl = torch.sum(kl, dim=(1, 2, 3))
    kl_loss = torch.mean(kl, dim=0)

    recon_x, _ = self.decode(z2)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / N

    weighted_kl_loss = self.kl_weight * kl_loss

    neg_elbo = recon_loss + weighted_kl_loss

    return neg_elbo, recon_loss, kl_loss

  def sample(self, num_samples):
    z1 = torch.randn((num_samples, 12, 2, 2), device=next(self.parameters()).device)
    with torch.no_grad():
      z2_mean = self.prior(z1)
      z2_var = torch.ones_like(z2_mean)
      z2 = self.reparameterize(z2_mean, z2_var)

      x, _ = self.decode(z2)
    return x