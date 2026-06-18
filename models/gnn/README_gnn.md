## Graph Neural Network

### Structure

This portion of the repository contains code for training the graph neural network. The directory contains:

`pdb_files` 

- pdb files corresponding to the spatial structure of each antibody
- Named according to the antibody ID in the Ginko provided data

`gnn_models.py` : GNN architecture

- Contains a class for a flexible GNN which takes in a config file for a sweep
- Reads in config, builds convolutional layers (message passing, batch normalization, dropout), pooling layers, and MLP layers
- Method for forward pass, required by torch
- Utility module `build_conv` automatically builds a convolutional layer using either NNConv or GENConv

`gnn_utils.py` : GNN utilities

- Concatenate training and prediction dataframes, divide into train and test based on hierarchical clustering folds and parse corresponding pdb files (`load_gnn_train_test`, `get_pdbs_and_targets`)
- Load residues from the pdb files and return antibody residues and coordinates (`load_pdb_residues`)
- Calculate pairwise features for edge attributes (`calculate_pairwise_features`)
- Build graph and return as a torch tensor (`build_graph`)
- Antibody graph class to convert train and test graphs into usable representation for torch DataLoader (`AntibodyGraphDataset`)
- Model evaluation tracking for wandb sweep integration (`ModelEvalTracker`)

`gnn_config.yaml` : Config file

- Defines hyperparameters relevant to individual runs, modify these to change individual run values

`gnn_train.py` : Main module

- Sets up training and validation loops for the `FlexibleGNN` using torch
- Integrated with wandb to perform hyperparameter sweep and/or document individual runs
- Takes in config file, target, and datafiles and performs an individual cross validation run

### Code Usage

To perform an individual cross validation run, simply update the hyperparameters to the desired values within `gnn_config.yaml` and run `python gnn_train.py`. This will automatically log the run and associated hyperparameters and artifacts in wandb. To modify which target this is run for, simply change the value of target within the main call of `gnn_train` to the string which corresponds to your desired target. 

To run a sweep, first ensure the hyperparameter space within `wandb_sweep.yaml` is defined as expected, then run the command `wandb sweep --project=antibody_gnn wandb_sweep.yaml`. This will output a wandb agent command which you can run verbatim on the command line to begin the sweep.  