import torch
import torch.nn as nn


class MixtureOfLogistics(nn.Module):
  def __init__(self, d, num_mix):
    super().__init__()
    self.d = d
    self.mix_scores = nn.Parameter(torch.randn(num_mix))
    self.means = nn.Parameter(torch.randn(num_mix) * d)  # *d for [0, 1] -> [0, d]
    self.scales_log = nn.Parameter(torch.randn(num_mix))  # use log for keep positivity

  def _prob(self, x):
    x = x.view(-1, 1)
    scales = torch.exp(self.scales_log)

    sigmoid_plus = torch.sigmoid((x + 0.5 - self.means) / scales)
    sigmoid_minus = torch.sigmoid((x - 0.5 - self.means) / scales)

    # process the edge cases for x=0 or x=d-1
    # # inplace operations
    # sigmoid_minus[x == 0] = 0
    # sigmoid_plus[x == self.d-1] = 1
    sigmoid_minus = torch.where(x == 0, torch.zeros_like(sigmoid_minus), sigmoid_minus)  # sigmoid(-inf)=0
    sigmoid_plus = torch.where(x == self.d-1, torch.ones_like(sigmoid_plus), sigmoid_plus)  # sigmoid(inf)=1

    sigmoid_diff = sigmoid_plus - sigmoid_minus
    mix_weights = torch.softmax(self.mix_scores, dim=0)
    # (N, num_mix) @ (num_mix,) -> (N,)
    probs = sigmoid_diff @ mix_weights
    return probs

  def loss(self, x):
    probs = self._prob(x)
    loss = torch.mean(-torch.log(probs))  # nll loss
    return loss
  
  def distribution(self):
    with torch.no_grad():
      return self._prob(torch.arange(self.d, device=self.means.device))