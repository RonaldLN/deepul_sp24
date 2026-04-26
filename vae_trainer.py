import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm


class VAETrainer:
  def __init__(self, model, train_dataset, test_dataset, batch_size,
              lr, epochs, lr_scheduler=None, device='cpu'):
    self.model = model
    self.device = device
    self.batch_size = batch_size
    self.epochs = epochs
    self.lr_scheduler = lr_scheduler

    self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    self.test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    self.opt = Adam(model.parameters(), lr=lr)

    self.epoch = 0
    self.train_losses = []
    self.test_losses = []

  def train_val(self, is_train):
    device = self.device
    if is_train:
      self.model.train()
      data_loader = self.train_loader
    else:
      self.model.eval()
      data_loader = self.test_loader
    losses, total_loss, total_num, data_bar = [], [0.0, 0.0, 0.0], 0, tqdm(data_loader)
    
    with (torch.enable_grad() if is_train else torch.no_grad()):
      for data in data_bar:
        data_size = data[0].size(0)
        data = tuple(d.to(device) for d in data)
        neg_elbo, recon_loss, kl_loss = self.model.loss(*data)  # (neg_elbo, recon_loss, kl_loss)
        losses.append([neg_elbo.item(), recon_loss.item(), kl_loss.item()])

        if is_train:
          self.opt.zero_grad()
          neg_elbo.backward()
          self.opt.step()

          if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        total_num += data_size
        total_loss[0] += neg_elbo.item() * data_size
        total_loss[1] += recon_loss.item() * data_size
        total_loss[2] += kl_loss.item() * data_size
        data_bar.set_description(f'{"Train" if is_train else "Test"} Epoch [{self.epoch}/{self.epochs}] -ELBO: {neg_elbo.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})')

    total_loss = [total_loss / total_num for total_loss in total_loss]
    return losses, total_loss

  def train(self):
    device = self.device
    self.model.to(device)

    _, init_total_loss = self.train_val(is_train=False)
    self.test_losses.append(init_total_loss)

    for epoch in range(1, self.epochs + 1):
      self.epoch = epoch
      train_losses, _ = self.train_val(is_train=True)
      self.train_losses.extend(train_losses)

      _, test_total_loss = self.train_val(is_train=False)
      self.test_losses.append(test_total_loss)