import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm


class CycleGANTrainer:
  def __init__(self, model, train_dataset1, train_dataset2, batch_size,
               lr, epochs, betas=(0.5, 0.999), device='cpu'):
    self.model = model.to(device)
    self.device = device
    self.batch_size = batch_size
    self.epochs = epochs

    self.train_loader1 = DataLoader(train_dataset1, batch_size, shuffle=True)
    self.train_loader2 = DataLoader(train_dataset2, batch_size, shuffle=True)

    self.d1_opt = Adam(model.discriminator1.parameters(), lr=lr, betas=betas)
    self.d2_opt = Adam(model.discriminator2.parameters(), lr=lr, betas=betas)
    self.g1_opt = Adam(model.generator1.parameters(), lr=lr, betas=betas)
    self.g2_opt = Adam(model.generator2.parameters(), lr=lr, betas=betas)

    self.epoch = 0

  def train_one_epoch(self):
    self.model.train()
    self.epoch += 1
    device = self.device
    total_d1_loss, total_d2_loss, total_num = 0.0, 0.0, 0
    data_bar = tqdm(zip(self.train_loader1, self.train_loader2), total=len(self.train_loader1))
    for data1, data2 in data_bar:
      data_size = data1[0].size(0)
      data1 = data1[0].to(device)  # x
      data2 = data2[0].to(device)  # y

      translated_y = self.model.generator1(data1)  # G(x)
      translated_x = self.model.generator2(data2)  # F(y)

      d1_loss = self.model.discriminator_loss(data1, translated_x, 1)  # Loss_D_X
      d2_loss = self.model.discriminator_loss(data2, translated_y, 2)  # Loss_D_Y

      self.d1_opt.zero_grad()
      d1_loss.backward()
      self.d1_opt.step()

      self.d2_opt.zero_grad()
      d2_loss.backward()
      self.d2_opt.step()

      g_loss = self.model.generator_loss(data1, data2, translated_y, translated_x)

      self.g1_opt.zero_grad()
      self.g2_opt.zero_grad()
      g_loss.backward()
      self.g1_opt.step()
      self.g2_opt.step()

      total_num += data_size
      total_d1_loss += d1_loss.item() * data_size
      total_d2_loss += d2_loss.item() * data_size
      data_bar.set_description(f'Train Epoch [{self.epoch}/{self.epochs}] D_X Loss: {total_d1_loss / total_num:.4f}, D_Y Loss: {total_d2_loss / total_num:.4f}')

  def train(self):
    for _ in range(self.epochs):
      self.train_one_epoch()