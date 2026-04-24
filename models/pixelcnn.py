import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class MaskedConv2d(nn.Conv2d):
  def __init__(self, mask_type, in_channels, out_channels, kernel_size,
               stride=1, padding=0, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride,
                     padding, bias=bias)
    mask = torch.ones_like(self.weight, dtype=torch.long)
    k_center = kernel_size // 2
    mask[:, :, k_center+1:] = 0  # Mask out rows below the center
    if mask_type == 'A':
      mask[:, :, k_center, k_center:] = 0  # Mask out the center and columns to the right
    elif mask_type == 'B':
      mask[:, :, k_center, k_center+1:] = 0  # Mask out columns to the right
    self.register_buffer("mask", mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super().forward(x)


class PixelCNN(nn.Module):
  def __init__(self, image_h, image_w, in_channels, n_filters, value_size):
    super().__init__()
    self.image_h = image_h
    self.image_w = image_w
    self.in_channels = in_channels

    self.layers = nn.ModuleList([
      MaskedConv2d('A', in_channels, n_filters, 7, padding=3),
      *[MaskedConv2d('B', n_filters, n_filters, 7, padding=3) for _ in range(5)],
      MaskedConv2d('B', n_filters, n_filters, 1, padding=0)
    ])
    out_channels = in_channels * value_size
    self.logits_layer = MaskedConv2d('B', n_filters, out_channels, 1, padding=0)

  def forward(self, x):
    for layer in self.layers:
      x = F.relu(layer(x))
    logits = self.logits_layer(x)
    return logits
  
  def loss(self, x):
    N, C, H, W = x.shape
    logits = self.forward(x)
    logits = logits.view(N, -1, C, H, W)
    loss = F.cross_entropy(logits, x.long())  # compute on dim=1
    return loss

  def samples(self, num_samples):
    C, H, W = self.in_channels, self.image_h, self.image_w
    x = torch.zeros(num_samples, C, H, W, device=self.logits_layer.weight.device)
    with torch.no_grad():
      for i in tqdm(range(H), desc="Generating samples (in rows)"):
        for j in range(W):
          logits = self.forward(x)
          logits_next = logits[:, :, i, j]  # (num_samples, C*value_size)
          logits_next = logits_next.view(num_samples*C, -1)  # (num_samples*C, value_size)
          probs = torch.softmax(logits_next, dim=1)
          next = torch.multinomial(probs, num_samples=1)  # (num_samples*C,)
          next = next.view(-1, C)  # (num_samples, C)
          x[:, :, i, j] = next
    return x