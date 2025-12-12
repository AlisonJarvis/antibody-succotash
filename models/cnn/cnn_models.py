import torch
import logging

logger = logging.Logger(__name__)
file_handler = logging.FileHandler('cnn_model.txt', mode='a')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ConvolutionLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        channels=8,
        dropout=0.5,
        stride=2,
        kernel_size=6,
        n_layers = 2,
        use_residuals = False
    ):
        if n_layers < 1:
            raise ValueError("Must have at least 1 convolution layer")
        self.use_residuals = use_residuals

        super().__init__()
        self.conv_layer_first = torch.nn.Sequential(
            *[
            torch.nn.LazyConv1d(
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            torch.nn.Dropout(dropout),
            torch.nn.PReLU()
        ])

        self.conv_layers_remaining = torch.nn.ModuleList()
        for i in range(n_layers-1):
            self.conv_layers_remaining.append(
                torch.nn.Sequential(*[
                    torch.nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=stride
                    ),
                    torch.nn.Dropout(dropout),
                    torch.nn.PReLU()
                ])
            )
        
        self.pooling_layer = torch.nn.MaxPool1d(
                kernel_size=kernel_size,
                stride=stride,
            )

    def forward(self, input):
        if self.use_residuals:
            x = input + self.conv_layer_first(input)
            for layer in self.conv_layers_remaining:
                x = layer(x) + x
        else:
            x = self.conv_layer_first(input)
            for layer in self.conv_layers_remaining:
                x = layer(x)   
        x = self.pooling_layer(x)
        return x         

class ConvModelFullVector(torch.nn.Module):
    def __init__(
        self,
        dropout=0.5,
        channels_log=3,
        n_conv_layers=2,
        kernel_size=6,
        stride=2
    ):
        # Handle case where channels_log < n_conv_layers
        if n_conv_layers > channels_log:
            raise ValueError(f"channels_log ({channels_log}) cannot be less than n_conv_layers ({n_conv_layers}).")
        
        super().__init__()

        self.mean_bias = 2.5

        # Set channels to 2^channels_log - we use log2 because 
        # channels will halve with each additional layer
        self.channels = 2**channels_log
        self.dropout = dropout
        self.n_conv_layers = n_conv_layers

        self.conv1 = torch.nn.Sequential(
            *[
                ConvolutionLayer(
                    dropout=dropout,
                    kernel_size=kernel_size,
                    stride=stride,
                    channels=self.channels//(2**i) # Halve channel size each time
                    )
                for i in range(n_conv_layers)
            ]
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.LazyLinear(8),
            torch.nn.Dropout(dropout),
            torch.nn.GELU(),
            torch.nn.Linear(8, 8),
            torch.nn.Dropout(dropout),
            torch.nn.GELU(),
        )

        # Module for generating variance estimate
        # We effectively cap it at 1 using a torch.sigmoid layer
        self.variance_module = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

        self.mean_module = torch.nn.Sequential(
            torch.nn.Linear(8, 1)
        )

    def forward(self, residue_features_sequence, subtype, gravy):
        #logger.debug(residue_features_sequence.shape)
        x = self.conv1(residue_features_sequence)
        #logger.debug(x.shape)
        x = x.flatten(start_dim=1)
        #logger.debug(x.shape)
        x = torch.cat([x, subtype], dim=-1)
        #logger.debug(x.shape)
        x = self.mlp(x)
        #logger.debug(x.shape)

        # Concatenate the mean and variance predictions 
        x = torch.cat([self.mean_module(x), self.variance_module(x)], dim=-1)
        #logger.debug(x.shape)
        # We set a base bias of 2.5 
        x[:, 0] += 2.5
        return torch.squeeze(x)