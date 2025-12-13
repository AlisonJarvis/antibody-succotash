# CNN

This directory contains all the code pertaining to the CNN model. There are 4 main files here. 

* `cnn_models.py` contains the model definition, and can be used in a standalone manner, if desired.

If a wandb account is set up, the following can be used:
* a single train run can be executed with `cnn_train.py`
* a sweep can be run using `cnn_sweep.py`

Lastly, a collection of helper functions may be found in `cnn_utils.py`

## Setup
To get started, ensure you have a python environment with the `requirements.txt` in the root of this repository installed. Then, run the `add_base_csv_to_wandb.py` script in `/data`, as well as the `add_embeddings_to_wandb.py` and `add_features_to_wandb.py` scripts in `/data_utils`. These will add the required artifacts to wandb.

## Running a sweep
To run a sweep, edit the sweep config found in `cnn_sweep.py` as desired, then run as a python script.