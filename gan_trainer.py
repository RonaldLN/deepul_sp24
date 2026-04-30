import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm


class GANTrainer:
  def __init__(self, model, k, train_dataset, batch_size,
              lr, epochs, betas=(0.9, 0.999), device='cpu'):
    self.model = model.to(device)
    self.k = k
    self.device = device
    self.batch_size = batch_size
    self.epochs = epochs

    self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    self.d_opt = Adam(model.discriminator.parameters(), lr=lr, betas=betas)
    self.g_opt = Adam(model.generator.parameters(), lr=lr, betas=betas)

    self.epoch = 0
    self.train_losses = []

  def train_one_epoch(self):
    self.epoch += 1
    device = self.device
    total_loss, total_num, data_bar = 0.0, 0, tqdm(self.train_loader)
    for data in data_bar:
      data_size = data[0].size(0)
      data = tuple(d.to(device) for d in data)

      for _ in range(self.k):
        d_loss = self.model.discriminator_loss(*data)
        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()
      self.train_losses.append(d_loss.item())

      g_loss = self.model.generator_loss(data_size)
      self.g_opt.zero_grad()
      g_loss.backward()
      self.g_opt.step()

      total_num += data_size
      total_loss += d_loss.item() * data_size
      data_bar.set_description(f'Train Epoch [{self.epoch}/{self.epochs}] Discriminator Loss: {total_loss / total_num:.4f}')

  def train(self):
    for _ in range(self.epoch, self.epochs):
      self.train_one_epoch()