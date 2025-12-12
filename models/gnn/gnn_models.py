# Imports
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    NNConv,
    GENConv,
    MLP,
    global_mean_pool,
    global_add_pool,
    global_max_pool
)
from torch_geometric.nn.aggr import Aggregation
from gnn_utils import global_mean_max_pool
from typing import Union, Optional

class NNGENConv(GENConv):
    def __init__(       
        self,
        in_channels: Union[int, tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, list[str], Aggregation]] = 'softmax',
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: str = 'batch',
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_dim: Optional[int] = None,
        edge_nn = MLP,
        edge_nn_kwargs: dict = {"num_layers": 2, "hidden_channels": 32},
        **kwargs
    ):
        super().__init__(in_channels, out_channels, aggr, t, learn_t, p, learn_p, msg_norm, learn_msg_scale, norm, num_layers, expansion, eps, bias, edge_dim, **kwargs)
        if edge_nn is not None:
            self.lin_edge = edge_nn(
                in_channels=edge_dim,
                out_channels=out_channels,
                bias=bias,
                **edge_nn_kwargs
            )


########## Function to build convolution layer ###########
def build_conv_layer(conv_type, in_dim, out_dim, edge_dim=None):

    ####### NNConv #########
    if conv_type == "nnconv":
        # MLP that maps edge attributes to weight matrix (flattened)
        edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim * out_dim)
        )
        # Creates NNConv layer
        return NNConv(
            in_channels=in_dim,
            out_channels=out_dim,
            nn=edge_mlp,
            aggr='mean'
        )

    ######## GENConv ##########
    elif conv_type == "mlpgenconv":
        # Define the GENConv layer
        # GENConv handles edge attributes automatically if passed in
        return NNGENConv(
            in_dim,
            out_dim,
            aggr='softmax',          # tends to be best for molecular type graphs
            t=1.0,                   # temperature parameter
            learn_t=True,            # learnable temperature
            learn_p=True,            # learnable power for message normalization
            msg_norm=True,
            norm='layer',
            learn_msg_scale=True,
            edge_dim=edge_dim        # tells GENConv how to use edges
        )
    ######## GENConv ##########
    elif conv_type == "genconv":
        # Define the GENConv layer
        # GENConv handles edge attributes automatically if passed in
        return GENConv(
            in_dim,
            out_dim,
            aggr='softmax',          # tends to be best for molecular type graphs
            t=1.0,                   # temperature parameter
            learn_t=True,            # learnable temperature
            learn_p=True,            # learnable power for message normalization
            msg_norm=True,
            norm='layer',
            learn_msg_scale=True,
            edge_dim=edge_dim        # tells GENConv how to use edges
        )

    else:
        raise ValueError(
            f"Unknown conv_type '{conv_type}', currently only recognizes 'nnconv', 'genconv' or 'mlpgenconv'."
        )


############ Flexible GNN class (w/ input config) ##############
class FlexibleGNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Reads parameters from input config, with defaults

        # Convolution layer config params
        in_dim        = config["input_dim"] # input dimension
        hidden_dim    = config.get("hidden_dim", 64) # hidden dimension
        conv_layers   = config.get("num_conv_layers", 3) # convolutional layers
        conv_type     = config.get("conv_type", "nnconv") # convolutional type
        edge_dim      = config.get("edge_dim", None) # edge dimension
        dropout       = config.get("dropout", 0.0) # dropout
        use_batchnorm = config.get("batchnorm", False) # whether to normalize batches

        # Dense layer config params
        mlp_layers    = config.get("num_mlp_layers", 2) # number of layers in MLP
        mlp_hidden    = config.get("mlp_hidden_dim", 128) # hidden dimension of MLP
        out_dim       = config.get("output_dim", 1) # Output dimension, always 1 for regression
        pooling       = config.get("pooling", "mean").lower() # pooling type, mean, min, max, attn

        # Chooses pooling method
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        elif pooling == "mean-max":
            self.pool = global_mean_max_pool
        else:
            raise ValueError("pooling must be 'mean', 'max', 'add', or 'mean-max'")

        # Define dropout, batchnorm, and convolution type parameters
        self.dropout = nn.Dropout(dropout)
        self.use_batchnorm = use_batchnorm
        self.conv_type = conv_type.lower()

        # Sets convolutional layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Actually builds up convolution layers with correct dimensions
        prev_dim = in_dim
        for _ in range(conv_layers):
            conv = build_conv_layer(conv_type, prev_dim, hidden_dim, edge_dim)
            self.convs.append(conv)

            # Define whether to use batchnorm for each layer
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            prev_dim = hidden_dim

        # Build the mlp -> regression block
        mlp = []
        mlp_in = hidden_dim
        # Account for mean-max pool doubling output dim through concatenation
        if pooling == "mean-max":
            mlp_in *= 2

        # Create mlp layers, each has linear, relu, dropout
        for _ in range(mlp_layers - 1):
            mlp.append(nn.Linear(mlp_in, mlp_hidden))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
            mlp_in = mlp_hidden

        # Adds the final regression layer to output dimension
        mlp.append(nn.Linear(mlp_in, out_dim))
        self.mlp = nn.Sequential(*mlp)

    ########### GNN forward pass ##################
    def forward(self, x, edge_index, batch, edge_attr):
        
        # Iterate through convolution layers
        for i, conv in enumerate(self.convs):
            # Actual convolution layer (NNConv or GENConv)
            x = conv(x, edge_index, edge_attr)
            # ReLU layer
            x = F.relu(x)
            # Batchnorm, if using normalization
            if self.use_batchnorm:
                x = self.bns[i](x)
            # Dropout layer
            x = self.dropout(x)

        # Pooling - converts variable numbers of nodes to graph embedding (fixed size)
        x = self.pool(x, batch)

        # Returns regression mlp layer
        return self.mlp(x) + 2.5
