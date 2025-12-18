import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .analysis import costhetastar


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


class DataNorm(nn.Module):
    def __init__(self, input_dim, mean=None, stddev=None):
        """
        Do the normalization: (x - mean) / stddev.

        :param input_dim: Dimension of the input features.
        :param mean:      Tensor of means for each feature.
        :param stddev:    Tensor of standard deviations for each feature.

        Note: If mean and stddev are provided, the normalization layer's weights
              and biases are set to perform the normalization and are made non-trainable.
              If mean and stddev are None, it is assumed, that the weights are set externally.
              This is needed for testing the model after loading from disk.
        """
        super().__init__()
        self.normalization        = nn.Linear(input_dim, input_dim)
        # Change the weights and biases to perform normalization and make them non-trainable.
        if not (mean is None and stddev is None):
            self.normalization.weight = nn.Parameter(torch.eye(input_dim) / stddev.unsqueeze(0), requires_grad=False)
            self.normalization.bias   = nn.Parameter(-mean / stddev, requires_grad=False)

    def forward(self, x):
        return self.normalization(x)


class FFNN_BatchNorm(nn.Module):
  def __init__(self, input_dim, width=200, stat_norm: dict = None, external_stat: bool = False):
    """
    Setup up a feedforward neural network with batch normalization,
    ELU activation functions, and optional input data normalization. The network
    consists of an input block, linear blocks (with and without dropout), and an
    output block.

    Args:
        input_dim (int): The dimensionality of the input features.
        width (int, optional): The width (number of neurons) of the hidden layers.
            Defaults to 200.
        stat_norm (dict, optional): Dictionary containing normalization statistics
            with 'mean' and 'stddev' keys for input data normalization. If None,
            no normalization is applied. Defaults to None.
        external_stat (bool, optional): If True, the normalization weights are
            expected to be set externally (e.g., during testing). Defaults to False.

    Note:
        The network uses ELU activation functions throughout, batch normalization
        on each layer, and includes a dropout layer (p=0.4) in one of the linear
        blocks for regularization. The output layer has no activation function.
    """
    super().__init__()
    self.input_dim = input_dim
    self.width     = width

    # Normalise input data if wished:
    if stat_norm is not None:
        self.data_norm = DataNorm(input_dim, stat_norm['mean'], stat_norm['stddev'])
    elif external_stat:
        self.data_norm = DataNorm(input_dim)  # Weights to be set externally.
    else:
        self.data_norm = nn.Identity()

    self.activation = nn.ELU()
    # self.activation = nn.Tanh()

    self.input_block = nn.Sequential(
      nn.BatchNorm1d(input_dim),
      nn.Linear(input_dim, width),
      nn.ELU()
    )

    self.linear_block = nn.Sequential(
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ELU(),
    )

    self.linear_block_drop = nn.Sequential(
      nn.BatchNorm1d(width),
      nn.Linear(width, width),
      nn.ELU(),
      nn.Dropout(p=0.4)
    )

    # Output layer:
    self.out_block = nn.Sequential(nn.BatchNorm1d(width), nn.Linear(width, 1))

    # No/ identity activation function for output layer:
    self.activ_output = nn.Identity()

  def forward(self, x):
    # Normalise input data if wished:
    x = self.data_norm(x)
    out = self.input_block(x)

    residual = out
    out = self.linear_block_drop(out)
    out = self.linear_block_drop(out) + residual  # Residual connection

    residual = out
    out = self.linear_block_drop(out)
    out = self.linear_block_drop(out) + residual  # Residual connection

    out = self.out_block(out)
    out = self.activ_output(out)
    return out


class FFNN_paper(nn.Module):
  def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3, stat_norm: dict = None, external_stat: bool = False):
    """
    Initialize a multi-layer perceptron neural network.
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output layer. Defaults to 1.
        emb_dim (list, optional): List of embedding dimensions for hidden layers.
            Defaults to [1000] * 3.
        stat_norm (dict, optional): Dictionary containing normalization statistics
            with 'mean' and 'stddev' keys for input data normalization. If None,
            no normalization is applied. Defaults to None.
    """
    super().__init__()
    self.input_dim  = input_dim
    self.output_dim = output_dim
    self.emb_dim    = emb_dim
    # FIXME: When I use the following definition and put them into a nn.Sequential,
    #        I get multiple ReLUs directly after each other in the graph. Why?
    # self.activation = nn.ReLU()
    # self.activation = nn.Tanh()

     # Normalise input data if wished:
    if stat_norm is not None:
        self.data_norm = DataNorm(input_dim, stat_norm['mean'], stat_norm['stddev'])
    elif external_stat:
        self.data_norm = DataNorm(input_dim)  # Weights to be set externally.
    else:
        self.data_norm = nn.Identity()

    # Multilayer Perceptron block:
    self.input_block = nn.Sequential(
      # nn.BatchNorm1d(self.input_dim),
      nn.Linear(self.input_dim, self.emb_dim[0]),
      nn.ReLU(),
    )

    self.hidden_block = nn.Sequential(
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

    # No/ identity activation function for output layer:
    self.activ_output = nn.Identity()

  def forward(self, x):
    x = self.data_norm(x)
    out = self.input_block(x)
    out = self.hidden_block(out)
    out = self.out_block(out)
    out = self.activ_output(out)
    return out

class FFNN_paper_163264(FFNN_paper):
    """
    Same as FFNN_paper but with emb_dim = [16, 32, 64].
    """
    def __init__(self, input_dim, output_dim = 1, *args, **kwargs):
        super().__init__(input_dim, output_dim, emb_dim = [16, 32, 64], *args, **kwargs)

class FFNN_paper_BatchNorm(FFNN_paper):
    """
    Same as FFNN_paper but with BatchNorm in input block.
    """
    def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3, *args, **kwargs):
        super().__init__(input_dim, output_dim, emb_dim, *args, **kwargs)
        # Override input block to include BatchNorm
        self.input_block = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.emb_dim[0]),
            nn.ReLU(),
        )

class FFNN_paper_nextraLayers(FFNN_paper):
    """
    Same as FFNN_paper but with `n_extra_layers` additional hidden layers.
    Each extra layer consists of a linear layer followed by an activation function.
    """
    def __init__(self, input_dim, output_dim=1, n_extra_layers=2, *args, **kwargs):
        super().__init__(input_dim, output_dim, emb_dim=[1000] * 3, *args, **kwargs)

        def calculate_new_width(input_dim: int, old_hidden_layers: int, new_hidden_layers: int, old_width: int) -> int:
            nparams = (((2 + input_dim + old_hidden_layers + old_hidden_layers * old_width) * old_width) + 1)

            new_width = -2 - input_dim - new_hidden_layers
            new_width += np.sqrt(4 + input_dim**2 + new_hidden_layers**2 + 2 * input_dim * (2 + new_hidden_layers) + 4 * new_hidden_layers * nparams)
            new_width /= (2 * new_hidden_layers)

            # new_width = - (2 + input_dim + new_hidden_layers - np.sqrt(4 + input_dim**2 + new_hidden_layers**2 + 2 * input_dim * (2 + new_hidden_layers) + 4 * new_hidden_layers * nparams)) / (2 * new_hidden_layers)

            return int(new_width)

        # Dynamically calculate the new width to keep the total number of parameters approximately constant
        new_width = calculate_new_width(self.input_dim, old_hidden_layers=4, new_hidden_layers=4 + n_extra_layers, old_width=1000)

        self.emb_dim = [new_width] * 3

        # Input block remains the same
        self.input_block = nn.Sequential(
            # nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.emb_dim[0]),
            nn.ReLU(),
        )

        # Dynamically create the extra layers
        extra_layers = [nn.Linear(self.emb_dim[0], self.emb_dim[0]), nn.ReLU()] * n_extra_layers

        # Hidden block with extra layers
        self.hidden_block = nn.Sequential(
            nn.Linear(self.emb_dim[0], self.emb_dim[1]),
            nn.ReLU(),
            nn.Linear(self.emb_dim[1], self.emb_dim[2]),
            nn.ReLU(),
            *extra_layers,  # Add the dynamically created extra layers here
            nn.Linear(self.emb_dim[2], self.emb_dim[1]),
            nn.ReLU(),
            nn.Linear(self.emb_dim[1], self.emb_dim[0]),
            nn.ReLU(),
        )

        # Output layer:
        self.out_block = nn.Linear(self.emb_dim[0], self.output_dim)

class FFNN_paper_2extraLayers(FFNN_paper_nextraLayers):
    """
    Same as FFNN_paper but with 2 extra hidden layers.
    """
    def __init__(self, input_dim, output_dim=1, *args, **kwargs):
        super().__init__(input_dim, output_dim, n_extra_layers=2, *args, **kwargs)

class FFNN_paper_4extraLayers(FFNN_paper_nextraLayers):
    """
    Same as FFNN_paper but with 4 extra hidden layers.
    """
    def __init__(self, input_dim, output_dim=1, *args, **kwargs):
        super().__init__(input_dim, output_dim, n_extra_layers=4, *args, **kwargs)

def minkowski_dot(p, q):
    """
    Computes Minkowski inner product for batches.
    p, q: tensors of shape [B, 4]
    metric diag = (1, -1, -1, -1)
    """
    return p[..., 3] * q[..., 3] - (p[..., 0:3] * q[..., 0:3]).sum(dim=-1)


class LorentzBaseLayer(nn.Module):
    """
    A layer that builds invariant and equivariant per-particle features
    from Lorentz 4-vectors.

    Takes raw 4-vectors [B, N, 4]
    Returns learned features [B, N, hidden_dim]
    """
    def __init__(self, n_4vectors, hidden_dim=128):
        super().__init__()
        self.n_4vectors = n_4vectors
        self.hidden_dim = hidden_dim

        # project invariant features
        self.inv_mlp = nn.Sequential(
            nn.Linear(1 + 1 + self.n_4vectors, hidden_dim),  # mass^2, norm(p), sum pairwise dot
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.eq_mlp = nn.Sequential(
            nn.Linear(self.n_4vectors, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, vectors):
        """
        vectors: [B, N, 4]: N = # of particles, each with (px, py, pz, E)
        returns: [B, N, hidden_dim]
        """
        B, N, _ = vectors.shape
        E = vectors[..., -1   ]
        P = vectors[...,   :-1]

        # mass^2 = E^2 - |p|^2  (shape [B, N])
        # Note: The invariant masses of the leptons are zero and thus not a meaningful feature.
        # mass2 = E**2 - (P**2).sum(dim=-1)

        if N == 4:
            angles = torch.stack(costhetastar(vectors), dim=-1)
        else:
          print("Warning: costhetastar not implemented for N != 4")
          raise NotImplementedError

        # norm(p) (shape [B, N])
        norm_p = torch.sqrt(torch.clamp((P**2).sum(dim=-1), min=1e-9))
        # pairwise Minkowski dot products (allocate on the same device as `vectors`)
        dot_mat = vectors.new_zeros(B, N, N)
        for i in range(N):
            for j in range(N):
                dot_mat[:, i, j] = minkowski_dot(vectors[:, i, :], vectors[:, j, :])

        # invariant feature vector per particle
        # "*torch.moveaxis(dot_mat, -1, 0) == dot_mat[..., 0], dot_mat[..., 1], dot_mat[..., 2], dot_mat[..., 3]"
        inv_feats = torch.stack([angles, norm_p, *torch.moveaxis(dot_mat, -1, 0)], dim=-1)  # [B, N, 2 + N] = [B, N, 6] for N = 4
        inv_feats = self.inv_mlp(inv_feats)  # [B, N, H]

        # equivariant projection of raw 4-vector
        eq_feats = self.eq_mlp(vectors)      # [B, N, H]

        return inv_feats + eq_feats

class FeatureBlock(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1   = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm1(x)
        y = F.gelu(self.fc1(y))
        y = self.norm2(y)
        y = self.dropout(F.gelu(self.fc2(y)))
        return x + y

class FourVectorAwareNet(nn.Module):
    """
    Full four-vector-aware network for 4 final-state particles.
    Predicts a single scalar (recommended: log(weight)).
    """
    def __init__(self, input_dim=16, hidden_dim=128, num_layers=3, predict_log=True, stat_norm: dict = None, external_stat: bool = False):
        super().__init__()
        # Input dimension: 1xsqrt(input_dim)xsqrt(input_dim)
        # Output dimension: output_dim
        assert int(input_dim**0.5)**2 == input_dim, "Input dimension must be a perfect square."
        self.n_4vectors = int(input_dim**0.5)

        # Normalise input data if wished:
        if stat_norm is not None:
            self.data_norm = DataNorm(input_dim, stat_norm['mean'], stat_norm['stddev'])
        elif external_stat:
            self.data_norm = DataNorm(input_dim)  # Weights to be set externally.
        else:
            self.data_norm = nn.Identity()

        self.predict_log = predict_log

        self.base = LorentzBaseLayer(self.n_4vectors, hidden_dim)

        self.blocks = nn.ModuleList([
            FeatureBlock(hidden_dim)
            for _ in range(num_layers)
        ])


        # Event-level aggregator (DeepSets)
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Output head
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B, 16] → reshape to [B, 4, 4]
        """

        # Normalise input data if wished:
        x = self.data_norm(x)

        x = x.view(x.size(0), 4, 4)  # 4 particles, each (E, px, py, pz)

        h = self.base(x)  # only applied once to 4-vectors

        for block in self.blocks:
            h = block(h)   # stackable residual layers

        pooled = h.mean(dim=1)       # permutation-invariant
        pooled = self.aggregator(pooled)
        out = self.output(pooled)
        # return out.squeeze(-1)
        return out


model_dict = {
    "FFNN_BatchNorm": FFNN_BatchNorm,
    "FFNN_paper": FFNN_paper,
    "FFNN_paper_163264": FFNN_paper_163264,
    "FFNN_paper_BatchNorm": FFNN_paper_BatchNorm,
    "FFNN_paper_2extraLayers": FFNN_paper_2extraLayers,
    "FFNN_paper_4extraLayers": FFNN_paper_4extraLayers,
    "FourVectorAwareNet": FourVectorAwareNet
}
