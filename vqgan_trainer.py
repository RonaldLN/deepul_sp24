import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm


class VQGANTrainer:
  def __init__(self, model, k, train_dataset, test_dataset, batch_size,
              lr, epochs, betas=(0.9, 0.999), device='cpu'):
    self.model = model.to(device)
    self.k = k
    self.device = device
    self.batch_size = batch_size
    self.epochs = epochs

    self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    self.test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    self.vqvae_opt = Adam(model.vqvae.parameters(), lr=lr, betas=betas)
    self.d_opt = Adam(model.discriminator.parameters(), lr=lr, betas=betas)

    self.epoch = 0
    self.train_discriminator_losses = []
    self.train_l_pips_losses = []
    self.train_l2_recon_losses = []
    self.test_l2_recon_losses = []

  def train_val(self, is_train):
    device = self.device
    if is_train:
      self.model.train()
      data_loader = self.train_loader
    else:
      self.model.eval()
      data_loader = self.test_loader
    total_l2_loss, total_num, data_bar = 0.0, 0, tqdm(data_loader)

    with (torch.enable_grad() if is_train else torch.no_grad()):
      for data in data_bar:
        data_size = data[0].size(0)
        x = data[0].to(device)

        vq_loss, l2_loss, recon_x = self.model.vqvae.loss(x)
        if is_train:
          for _ in range(self.k):
            d_loss = self.model.discriminator_loss(x, recon_x)
            self.d_opt.zero_grad()
            d_loss.backward()
            self.d_opt.step()
          self.train_discriminator_losses.append(d_loss.item())

          vqvae_loss, perceptual_loss, l2_loss = self.model.vqvae_loss(x, recon_x, vq_loss, l2_loss)
          self.vqvae_opt.zero_grad()
          vqvae_loss.backward()
          self.vqvae_opt.step()

          self.train_l_pips_losses.append(perceptual_loss.item())
          self.train_l2_recon_losses.append(l2_loss.item())
        else:
          d_loss = self.model.discriminator_loss(x, recon_x)
          vqvae_loss, perceptual_loss, l2_loss = self.model.vqvae_loss(x, recon_x, vq_loss, l2_loss)

        total_num += data_size
        total_l2_loss += l2_loss.item() * data_size
        data_bar.set_description(f'{"Train" if is_train else "Test"} Epoch [{self.epoch}/{self.epochs}] Loss: {vqvae_loss.item():.4f} (Discriminator Loss: {d_loss.item():.4f}, L2 Loss: {l2_loss.item():.4f}, LPIPS Loss: {perceptual_loss.item():.4f})')

    return total_l2_loss / total_num

  def train(self):
    init_total_l2_loss = self.train_val(is_train=False)
    self.test_l2_recon_losses.append(init_total_l2_loss)

    for epoch in range(1, self.epochs + 1):
      self.epoch = epoch
      self.train_val(is_train=True)

      test_total_l2_loss = self.train_val(is_train=False)
      self.test_l2_recon_losses.append(test_total_l2_loss)