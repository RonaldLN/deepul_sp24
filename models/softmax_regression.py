import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxRegression(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.theta = nn.Parameter(torch.zeros(d))

  def loss(self, x):
    scores = self.theta.expand(x.shape[0], -1)
    return F.cross_entropy(scores, x)
  
  def distribution(self):
    with torch.no_grad():
      return torch.softmax(self.theta, dim=0)