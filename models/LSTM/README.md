# LSTM HIC Prediction
### Kayleen Nordyke 12/05/2025

Trains a PyTorch LSTM regressor to predict antibody HIC using:

    > Token-level transformer embeddings from ollieturnbull/p-IgGen
    > Engineered physiochemical features computed using BioPython
    > 5-fold CV using hierarchical_cluster_IgG_isotype_stratified_fold

## Inputs
Ensure that these CSVs are within data/
    > data/GDPa1_v1.2_sequences.csv
    > data/data/GDPa1_v1.2_20250814.csv

## Setup
Ensure that these dependencies are installed
    > pandas
    > numpy
    > scikit-learn
    > scipy
    > torch
    > transformers
    > tqdm
    > biopython

## Run
To run the script
    > python LSTM_model.py

Results are saved into a csv file titled "lstm_results.csv"

Included is a Python notebook, LSTM_demo.ipynb, which contains the same workflow as LSTM_model.py but is split into step-by-step cells for clarity and easier inspection of each stage.