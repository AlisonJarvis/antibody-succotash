import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import wandb
from cnn_train import main

from data_utils.feature_utils import create_features_from_raw_df

# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "spearman"},
    "parameters": {
        "lr": {"max": 1e-2, "min": 1e-7, "distribution": "log_uniform_values"},
        "epochs": {"max": 200, "min": 10, "distribution": "int_uniform"},
        "model_params": {
            "parameters": {
                "norm_layer": {"values":[True, False]},
                "residuals": {"values": [True]},
                "dropout": {"max": 0.5, "min": 0, "distribution": "uniform"},
                "kernel_size": {"max":9, "min": 6, "distribution": "int_uniform"},
                "stride": {"value": 1, "distribution":"constant"},
                "channels_log": {"max":6, "min":3, "distribution": "int_uniform"},
                "n_conv_layers": {"max":3, "min": 1, "distribution": "int_uniform"},
            }
        },
        "pca_reduction":  {"values": [8], "distribution": "categorical"},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,
        "eta": 2,
    }
}



# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Antibody Succotash")
wandb.agent(sweep_id, main)
