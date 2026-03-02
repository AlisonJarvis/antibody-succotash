# LSTM HIC Prediction
**Kayleen Nordyke — 12/05/2025**

Trains a PyTorch LSTM regressor to predict antibody HIC using:
- **Token-level transformer embeddings** from `ollieturnbull/p-IgGen`
- **Engineered physicochemical features** computed with **BioPython**
- **5-fold cross-validation** using `hierarchical_cluster_IgG_isotype_stratified_fold`

## Inputs
Ensure these CSVs are in `data/`:
- `data/GDPa1_v1.2_sequences.csv`
- `data/GDPa1_v1.2_20250814.csv`

## Setup
Install dependencies:
- pandas
- numpy
- scikit-learn
- scipy
- torch
- transformers
- tqdm
- biopython

## Run
```bash
python LSTM_model.py
```

## Outputs
Results are saved into a CSV file titled `lstm_results.csv`.

## Notebook
Included is a Python notebook, `LSTM_demo.ipynb`, which contains the same workflow as `LSTM_model.py` but is split into step-by-step cells for clarity and easier inspection of each stage.
