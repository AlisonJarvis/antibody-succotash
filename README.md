# Antibody Developability Prediction (Hydrophobicity)
This repository contains code for predicting antibody developability-related properties, with a focus on hydrophobicity measured by Hydrophobic Interaction Chromatography (HIC).

The project investigates whether antibody VH/VL sequences can be used to predict developability behavior using machine-learning models, and compares feature-based and sequence-based approaches.

## Project Overview
This project focuses on learning relationships between antibody sequence information and experimentally measured hydrophobicity.
Machine-learning models are trained to predict HIC values from antibody sequences using either engineered physicochemical descriptors or learned sequence representations.
The primary goal is to rank antibodies by relative hydrophobicity, rather than to precisely predict absolute values.

## Repository Structure
```
├── data/                    # raw CSV files and input datasets
├── data_utils/              # shared utilities (features, embeddings, scoring)
├── feature_engineering/     # notebooks and scripts for feature construction
├── models/                  # model implementations
├── tests/                   # basic unit tests
├── sandbox/                 # exploratory notebooks and scratch work
└── requirements.txt         # Python dependencies
```

## Data Description
The dataset originates from the Ginkgo Bioworks antibody developability benchmark.
Two primary input files are used:
- a sequence file containing antibody VH and VL sequences and metadata
- a property file containing experimentally measured developability values, including HIC
The files are linked using a shared antibody identifier. Although multiple developability-related measurements are available, this project focuses specifically on hydrophobicity as measured by HIC.


## Feature Engineering
Feature engineering in this project is centered around sequence-derived physicochemical descriptors, such as hydrophobicity, charge-related properties, aromaticity, and amino-acid composition.

Although the underlying features remain consistent across the project, they are explicitly organized into a hierarchical structure (Fv, chain, and CDR levels) only within the Random Forest models. This structured representation was introduced to facilitate systematic analysis and interpretation, including feature importance evaluation.

This design choice was motivated by exploratory analysis showing that individual sequence-derived descriptors exhibit limited linear correlation with HIC, motivating the use of structured feature representations and non-linear models.

General feature computation and exploratory analysis are supported by shared utilities in `data_utils/` and `feature_engineering/`, while the hierarchical feature organization is implemented as part of the Random Forest modeling workflow in `models/RF/`.


## Models
All model implementations are organized under the models/ directory.
Each subdirectory corresponds to a different modeling approach for predicting hydrophobicity from antibody sequences, framed as a regression problem and designed to compare feature-based and sequence-based methods.


### Random Forest (RF)
The Random Forest approach uses engineered physicochemical features derived from antibody sequences and is robust to non-linear feature interactions.
In this project, RF is the primary setting where features are explicitly organized and explored at multiple levels (Fv, chain, and CDR), enabling interpretability and feature importance analysis.

### LSTM
LSTM models treat antibody sequences as ordered amino-acid chains and learn representations directly from sequence order.
They are used to capture long-range dependencies that may influence hydrophobicity, particularly when combined with learned embeddings.

### GNN
GNN models aim to capture non-local interactions between residues that arise from the three-dimensional structure of antibodies.
By representing antibodies as graphs derived from predicted structures, this approach incorporates spatial relationships that are not accessible from sequence-based models alone.

### CNN
CNN models learn local sequence patterns from residue-level embeddings using convolution and pooling operations.
Multiple architectures and training hyperparameters were evaluated using cross-validation to assess how local sequence motifs contribute to hydrophobic behavior.

## Notes
Model evaluation emphasizes rank-based performance metrics, particularly Spearman correlation, using predefined cross-validation folds.Each model directory may include additional documentation specific to that approach.
The repository structure is intentionally modular to support future extensions.

