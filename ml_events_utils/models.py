import torch
import logging

from torch import nn
import torch.nn.functional as F
import numpy as np
from .analysis import costhetastar

logger = logging.getLogger(__name__)

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
  def __init__(self, input_dim, width=200, stat_norm: dict = None, external_stat: bool = False, *args, **kwargs):
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

class FFNN_BatchNorm_nextraLayers(FFNN_BatchNorm):
    def __init__(self, input_dim, n_extra_layers=2, *args, **kwargs):
        def calculate_new_width(input_dim: int, old_hidden_layers: int, new_hidden_layers: int, old_width: int) -> int:
            nparams = (((2 + input_dim + old_hidden_layers + old_hidden_layers * old_width) * old_width) + 1)

            new_width = -2 - input_dim - new_hidden_layers
            new_width += np.sqrt(4 + input_dim**2 + new_hidden_layers**2 + 2 * input_dim * (2 + new_hidden_layers) + 4 * new_hidden_layers * nparams)
            new_width /= (2 * new_hidden_layers)

            # new_width = - (2 + input_dim + new_hidden_layers - np.sqrt(4 + input_dim**2 + new_hidden_layers**2 + 2 * input_dim * (2 + new_hidden_layers) + 4 * new_hidden_layers * nparams)) / (2 * new_hidden_layers)

            return int(new_width)

        self.input_dim = input_dim
        self.n_extra_layers = n_extra_layers

        # Dynamically calculate the new width to keep the total number of parameters approximately constant
        new_width = calculate_new_width(self.input_dim, old_hidden_layers=4, new_hidden_layers=4 + self.n_extra_layers, old_width=200)

        super().__init__(self.input_dim, width=new_width, *args, **kwargs)

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

        for _ in range(self.n_extra_layers):  # Extra layers
            residual = out
            out = self.linear_block_drop(out)
            out = self.linear_block_drop(out) + residual  # Residual connection

        out = self.out_block(out)
        out = self.activ_output(out)
        return out

class FFNN_paper(nn.Module):
  def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3, stat_norm: dict = None, external_stat: bool = False, *args, **kwargs):
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

class FFNN_EMB_Selection(nn.Module):
  def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3, stat_norm: dict = None, external_stat: bool = False, *args, **kwargs):
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

    # The following hidden block takes the dimensions from emb_dim and first goes up and then down again. The number of layers is determined by the length of emb_dim.
    #  For example, if emb_dim = [100, 200, 400], the hidden block will have the following layers:
    #  Linear(100, 200) -> ReLU -> Linear(200, 400) -> ReLU -> Linear(400, 200) -> ReLU -> Linear(200, 100) -> ReLU
    hlayers = []
    for i in range(len(self.emb_dim) - 1):
      hlayers.append(nn.Linear(self.emb_dim[i], self.emb_dim[i+1]))
      hlayers.append(nn.ReLU())
    for i in range(len(self.emb_dim) - 1, 0, -1):
      hlayers.append(nn.Linear(self.emb_dim[i], self.emb_dim[i-1]))
      hlayers.append(nn.ReLU())

    self.hidden_block = nn.Sequential(*hlayers)

    # Output layer:
    self.out_block = nn.Linear(self.emb_dim[0], self.output_dim)

  def forward(self, x):
    x = self.data_norm(x)
    out = self.input_block(x)
    out = self.hidden_block(out)
    out = self.out_block(out)
    return out

class FFNN_EMB_Selection_BatchNorm(FFNN_EMB_Selection):
    """
    Same as FFNN_EMB_Selection but with BatchNorm in input block.
    """
    def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3, *args, **kwargs):
        super().__init__(input_dim, output_dim, emb_dim, stat_norm = None, external_stat = False, *args, **kwargs)
        # Override input block to include BatchNorm
        self.input_block = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.emb_dim[0]),
            nn.ReLU(),
        )

class FFNN_EMB_Selection_LayerNorm(FFNN_EMB_Selection):
    """
    Same as FFNN_EMB_Selection but with LayerNorm in input block.
    """
    def __init__(self, input_dim, output_dim = 1, emb_dim = [1000] * 3, *args, **kwargs):
        super().__init__(input_dim, output_dim, emb_dim, stat_norm = None, external_stat = False, *args, **kwargs)
        # Override input block to include LayerNorm
        self.input_block = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.emb_dim[0]),
            nn.ReLU(),
        )

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
        def calculate_new_width(input_dim: int, old_hidden_layers: int, new_hidden_layers: int, old_width: int) -> int:
            nparams = (((2 + input_dim + old_hidden_layers + old_hidden_layers * old_width) * old_width) + 1)

            new_width = -2 - input_dim - new_hidden_layers
            new_width += np.sqrt(4 + input_dim**2 + new_hidden_layers**2 + 2 * input_dim * (2 + new_hidden_layers) + 4 * new_hidden_layers * nparams)
            new_width /= (2 * new_hidden_layers)

            # new_width = - (2 + input_dim + new_hidden_layers - np.sqrt(4 + input_dim**2 + new_hidden_layers**2 + 2 * input_dim * (2 + new_hidden_layers) + 4 * new_hidden_layers * nparams)) / (2 * new_hidden_layers)

            return int(new_width)

        self.input_dim = input_dim
        self.n_extra_layers = n_extra_layers

        # Dynamically calculate the new width to keep the total number of parameters approximately constant
        new_width = calculate_new_width(self.input_dim, old_hidden_layers=4, new_hidden_layers=4 + self.n_extra_layers, old_width=1000)

        super().__init__(self.input_dim, output_dim, emb_dim=[new_width] * 3, *args, **kwargs)

        # Dynamically create the extra layers
        extra_layers = []
        for _ in range(n_extra_layers):
            extra_layers.append(nn.Linear(self.emb_dim[0], self.emb_dim[0]))
            extra_layers.append(nn.ReLU())

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

class FFNN_paper_nextraLayers_BatchNorm(FFNN_paper_nextraLayers):
    """
    Same as FFNN_paper_nextraLayers but with BatchNorm in input block.
    """
    def __init__(self, input_dim, output_dim = 1, n_extra_layers=2, *args, **kwargs):
        super().__init__(input_dim, output_dim, n_extra_layers=n_extra_layers, *args, **kwargs)
        # Override input block to include BatchNorm
        self.input_block = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.emb_dim[0]),
            nn.ReLU(),
        )

class FFNN_general(nn.Module):
    """
    A general feedforward neural network with a variable number of hidden layers and (constant)width.
    Each extra layer consists of a linear layer followed by an activation function.
    """
    def __init__(self, input_dim:int, width:int, n_hidden:int, output_dim=1, *args, **kwargs):
        super().__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim

        self.width = int(width)
        self.n_hidden = int(n_hidden)

        self.emb_dim = [self.width] * self.n_hidden

        self.input_block = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.emb_dim[0]),
            nn.ReLU(),
        )

        # Dynamically create the hidden layers
        hidden_layers = []
        for i in range(len(self.emb_dim) - 1):
            hidden_layers.append(nn.Linear(self.emb_dim[i], self.emb_dim[i + 1]))
            hidden_layers.append(nn.ReLU())

        # Hidden block
        self.hidden_block = nn.Sequential(*hidden_layers)

        # Output layer:
        self.out_block = nn.Linear(self.emb_dim[-1], self.output_dim)

    def forward(self, x):
        out = self.input_block(x)
        out = self.hidden_block(out)
        out = self.out_block(out)
        return out

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

    def __init__(self, n_4vectors, hidden_dim=128, normalize=False):
        super().__init__()
        self.n_4vectors = n_4vectors
        self.hidden_dim = hidden_dim

        # Prepare the normalization layers if normalize=True
        if normalize:
            self.inv_norm = nn.LayerNorm(1 + 1 + self.n_4vectors)
            # Normalize over the last layer of the input 4-vectors
            self.eq_norm = nn.LayerNorm(4)
            mpl_start = 4
        else:
            self.inv_norm = None
            self.eq_norm = None
            # but make sure it works the same in the not-normalize case
            mpl_start = self.n_4vectors

        # project invariant features
        self.inv_mlp = nn.Sequential(
            nn.Linear(1 + 1 + self.n_4vectors, hidden_dim),  # mass^2, norm(p), sum pairwise dot
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.eq_mlp = nn.Sequential(
            nn.Linear(mpl_start, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, vectors):
        """
        vectors: [B, N, 4]: N = # of particles, each with (px, py, pz, E)
        returns: [B, N, hidden_dim]
        """
        B, N, _ = vectors.shape
        E = vectors[..., -1]
        P = vectors[..., :-1]

        # mass^2 = E^2 - |p|^2  (shape [B, N])
        # Note: The invariant masses of the leptons are zero and thus not a meaningful feature.
        # mass2 = E**2 - (P**2).sum(dim=-1)

        if N == 4:
            angles = torch.stack(costhetastar(vectors), dim=-1)
        else:
            msg = f"Warning: costhetastar not implemented for N != 4 ({N=})"
            logger.error(msg)
            raise NotImplementedError(msg)

        # norm(p) (shape [B, N])
        norm_p = torch.sqrt(torch.clamp((P**2).sum(dim=-1), min=1e-9))
        # pairwise Minkowski dot products (allocate on the same device as `vectors`)
        dot_mat = vectors.new_zeros(B, N, N)
        for i in range(N):
            for j in range(N):
                dot_mat[:, i, j] = minkowski_dot(vectors[:, i, :], vectors[:, j, :])

        # invariant feature vector per particle
        # "*torch.moveaxis(dot_mat, -1, 0) == dot_mat[..., 0], dot_mat[..., 1], dot_mat[..., 2], dot_mat[..., 3]"
        inv_feats = torch.stack(
            [angles, norm_p, *torch.moveaxis(dot_mat, -1, 0)], dim=-1
        )  # [B, N, 2 + N] = [B, N, 6] for N = 4
        if self.inv_norm is not None:
            # Before going throught he MLPs, normalize the input
            inv_feats = self.inv_norm(inv_feats)
            vectors = self.eq_norm(vectors)

        inv_feats = self.inv_mlp(inv_feats)  # [B, N, H]

        # equivariant projection of raw 4-vector
        eq_feats = self.eq_mlp(vectors)  # [B, N, H]

        return inv_feats + eq_feats


# Implementatio of ParticleNet
# Refs
# > https://cms-ml.github.io/documentation/inference/particlenet.html
# > https://indico.cern.ch/event/980214/contributions/4413544/attachments/2277334/3868991/ParticleNeXt_ML4Jets2021_H_Qu.pdf
# > https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.conv.EdgeConv.html
class EdgeConv(nn.Module):
    """
    Edge convolution layer from https://arxiv.org/abs/1801.07829

    Cloud-of-particles version of applying a convolutional NN to a picture.
    Particles are treated as nodes in a graph, so we will look as particles in the context
    of all other particles in the event.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_feats, out_feats),
            nn.LayerNorm(out_feats),
            nn.GELU(),
            nn.Linear(out_feats, out_feats),
            nn.LayerNorm(out_feats),
            nn.GELU(),
        )

    def forward(self, x):
        # x.shape = [Batch, N particles, C (4... at the begining)]
        _, N, _ = x.shape

        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)
        # Construct edge features for complete graph: [B, N, N, 2*C]
        # Basically we pass through the 'position' of the particles
        # but concatenated to its distance to each other particle
        edge_input = torch.cat([x_i, x_j - x_i], dim=-1)

        # Apply now the edgeconv FFNN
        edge_features = self.mlp(edge_input)  # [B, N, N, out_feats]
        return edge_features.mean(dim=2)  # [B, N, out_feats]


class ParticleNet(nn.Module):
    """
    Implementation of the ParticleNet https://arxiv.org/pdf/1902.08570
    Basically Fig 2, but adding the LorentzBaseLayer before the input

    input (4-vectors)
    LorentzBaseLayer to generate an invariant set of <embed_dim> features
    <num_layers> EdgeConvs (each twice as wide as the previous one)
    output mpl made of:
        dense
        relu
        dropout
        dense
    """

    def __init__(
        self,
        input_dim=16,
        output_dim=1,
        embed_dim=128,
        num_layers=3,
        growing_edge=False,
        **kwargs,
    ):
        super().__init__()
        self.n_4vectors = int(input_dim**0.5)
        self.base = LorentzBaseLayer(
            n_4vectors=self.n_4vectors, hidden_dim=embed_dim, normalize=True
        )

        current_dim = next_dim = embed_dim

        tmp = []
        for _ in range(num_layers):
            if growing_edge:
                next_dim = current_dim * 2
            tmp.append(EdgeConv(current_dim, next_dim))
            current_dim = next_dim

        self.convs = nn.ModuleList(tmp)

        self.output_mpl = nn.Sequential(
            nn.Linear(current_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x):
        # taken from FourVectorAwareNet below
        # Each event has 16 numbers which are 4 particles
        # each with (E, px, py, pz)
        # x: [B, 16] → reshape to [B, 4, 4]
        x = x.view(x.size(0), self.n_4vectors, 4)
        h = self.base(x)
        for conv in self.convs:
            h = conv(h)
        return self.output_mpl(h.mean(dim=1))


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
        if stat_norm is not None or external_stat:
            logger.info("FourVectorAwareNet: Data normalization is not applied, since the model works on 4-vectors directly.")

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
    "FFNN_BatchNorm_2extraLayers": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_BatchNorm_nextraLayers(input_dim, n_extra_layers=2, *args, **kwargs),
    "FFNN_BatchNorm_4extraLayers": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_BatchNorm_nextraLayers(input_dim, n_extra_layers=4, *args, **kwargs),
    "FFNN_paper": FFNN_paper,
    "FFNN_paper_163264": FFNN_paper_163264,
    "FFNN_paper_BatchNorm": FFNN_paper_BatchNorm,
    "FFNN_paper_2extraLayers": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_paper_nextraLayers(input_dim, output_dim, n_extra_layers=2, *args, **kwargs),
    "FFNN_paper_4extraLayers": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_paper_nextraLayers(input_dim, output_dim, n_extra_layers=4, *args, **kwargs),
    "FFNN_paper_8extraLayers": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_paper_nextraLayers(input_dim, output_dim, n_extra_layers=8, *args, **kwargs),
    "FFNN_paper_2extraLayers_BatchNorm": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_paper_nextraLayers_BatchNorm(input_dim, output_dim, n_extra_layers=2, *args, **kwargs),
    "FFNN_paper_4extraLayers_BatchNorm": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_paper_nextraLayers_BatchNorm(input_dim, output_dim, n_extra_layers=4, *args, **kwargs),
    "FFNN_paper_8extraLayers_BatchNorm": lambda input_dim, output_dim=1, *args, **kwargs: FFNN_paper_nextraLayers_BatchNorm(input_dim, output_dim, n_extra_layers=8, *args, **kwargs),
    #
    "FFNN_EMB_512_256_128_64_32": lambda input_dim, output_dim=1, emb_dim=[512,256,128,64,32], *args, **kwargs: FFNN_EMB_Selection(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    "FFNN_EMB_1024_512_256_128_64_32": lambda input_dim, output_dim=1, emb_dim=[1024,512,256,128,64,32], *args, **kwargs: FFNN_EMB_Selection(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    "FFNN_EMB_1024_512_256_128_64": lambda input_dim, output_dim=1, emb_dim=[1024,512,256,128,64], *args, **kwargs: FFNN_EMB_Selection(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    "FFNN_EMB_32_64_128_256_512": lambda input_dim, output_dim=1, emb_dim=[32,64,128,256,512], *args, **kwargs: FFNN_EMB_Selection(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    "FFNN_EMB_32_64_128_256_512_1024": lambda input_dim, output_dim=1, emb_dim=[32,64,128,256,512,1024], *args, **kwargs: FFNN_EMB_Selection(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    #
    "FFNN_EMB_1024_512_256_128_64_BatchNorm": lambda input_dim, output_dim=1, emb_dim=[1024,512,256,128,64], *args, **kwargs: FFNN_EMB_Selection_BatchNorm(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    "FFNN_EMB_1024_512_256_128_64_LayerNorm": lambda input_dim, output_dim=1, emb_dim=[1024,512,256,128,64], *args, **kwargs: FFNN_EMB_Selection_LayerNorm(input_dim, output_dim=output_dim, emb_dim=emb_dim, *args, **kwargs),
    #
    "FourVectorAwareNet": FourVectorAwareNet,
    "FFNN_general": FFNN_general,
    "ParticleNet": ParticleNet,
    "ParticleNet_big": lambda input_dim, *args, **kwargs: ParticleNet(input_dim, *args, **kwargs),
    "ParticleNet_best": lambda input_dim, *args, **kwargs: ParticleNet(input_dim, *args, embed_dim=80, num_layers=2),
    "ParticleNet_growing": lambda input_dim, *args, **kwargs: ParticleNet(input_dim, embed_dim=64, growing_edge=True, *args, **kwargs),
}
