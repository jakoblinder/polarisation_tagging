import torch
from torch import nn


# Activation function for output layer:
# For binary classification tasks, common choices are:
# Use Sigmoid for binary classification.
# self.activ_output = nn.Sigmoid()
#
# For multi-class use Softmax.
# self.activ_output = nn.Softmax()

# For regression tasks, common choices are:
# ReLU
# self.activ_output = nn.ReLU()
# LeakyReLU
# self.activ_output = nn.LeakyReLU(negative_slope=0.01)
# ELU (exponential decay for negative inputs)
# self.activ_output = nn.ELU()


class FFNN_BatchNorm(nn.Module):
  def __init__(self, input_dim, width=1000):
    super().__init__()

    # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    # Multilayer Perceptron block:
    self.mlp_block = nn.Sequential(
      nn.BatchNorm1d(input_dim),
      nn.Linear(input_dim, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU()
    )

    # Output layer:
    self.out_block = nn.Sequential(nn.BatchNorm1d(width), nn.Linear(width, 1))

    # Activation function for output layer:
    self.activ_output = nn.ReLU()
    # Try ELU
    # self.activ_output = nn.ELU()

  def forward(self, x):
    out = self.mlp_block(x)
    out = self.out_block(out)
    out = self.activ_output(out)
    return out

class FFNN_BatchNorm_no_output(nn.Module):
  def __init__(self, input_dim, width=1000):
    super().__init__()

    # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    # Multilayer Perceptron block:
    self.mlp_block = nn.Sequential(
      nn.BatchNorm1d(input_dim),
      nn.Linear(input_dim, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ReLU()
    )

    # Output layer:
    self.out_block = nn.Sequential(nn.BatchNorm1d(width), nn.Linear(width, 1))

  def forward(self, x):
    out = self.mlp_block(x)
    out = self.out_block(out)
    return out


class FFNN_paper(nn.Module):
  def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.emb_dim = emb_dim

    # Multilayer Perceptron block:
    self.mlp_block = nn.Sequential(
      nn.Linear(self.input_dim, self.emb_dim[0]),
      nn.ReLU(),
      nn.Linear(self.emb_dim[0], self.emb_dim[1]),
      nn.ReLU(),
      nn.Linear(self.emb_dim[1], self.emb_dim[2]),
      nn.ReLU(),
      nn.Linear(self.emb_dim[2], self.emb_dim[1]),
      nn.ReLU(),
      nn.Linear(self.emb_dim[1], self.emb_dim[0]),
      nn.ReLU(),
    )

    # Output layer:
    self.out_block = nn.Linear(self.emb_dim[0], self.output_dim)

    # Activation function for output layer:
    self.activ_output = nn.ReLU()

  def forward(self, x):
    out = self.mlp_block(x)
    out = self.out_block(out)
    out = self.activ_output(out)
    return out


model_dict = {
    "FFNN_BatchNorm": FFNN_BatchNorm,
    "FFNN_BatchNorm_no_output": FFNN_BatchNorm_no_output,
    "FFNN_paper": FFNN_paper
}