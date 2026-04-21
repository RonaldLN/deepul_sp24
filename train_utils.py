import torch
from tqdm.auto import tqdm


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cpu'):
  is_train = train_optimizer is not None
  model.train() if is_train else model.eval()
  losses, total_loss, total_num, data_bar = [], 0.0, 0, tqdm(data_loader)
  with (torch.enable_grad() if is_train else torch.no_grad()):
    for data in data_bar:
      data = data[0].to(device)
      loss = model.loss(data)
      losses.append(loss.item())

      if is_train:
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

      total_num += data.size(0)
      total_loss += loss.item() * data.size(0)
      data_bar.set_description(f'{"Train" if is_train else "Test"} Epoch [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')
  
  return losses, total_loss / total_num


def train(model, train_optimizer, train_loader, test_loader, epochs, device='cpu'):
  train_losses, test_losses = [], []

  _, init_test_loss = train_val(model, test_loader, None, 0, epochs, device)
  test_losses.append(init_test_loss)

  for epoch in range(1, epochs + 1):
    train_epoch_losses, _ = train_val(model, train_loader, train_optimizer, epoch, epochs, device)
    train_losses.extend(train_epoch_losses)

    _, test_loss = train_val(model, test_loader, None, epoch, epochs, device)
    test_losses.append(test_loss)

  return train_losses, test_losses