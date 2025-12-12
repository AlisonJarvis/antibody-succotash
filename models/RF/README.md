# Random Forest (RF) – Hydrophobicity Prediction
This folder contains code for predicting HIC (Hydrophobic Interaction Chromatography) values from antibody sequences using a Random Forest regression model.
The model uses features derived from:
- the full Fv sequence (VH + VL),
- individual VH and VL chains,
- CDR regions (HCDR3 and LCDR1, based on AHo numbering),
with a strong focus on hydrophobicity-related physicochemical properties.

## 1. Folder Structure
```
models/RF/
├── RF_utils.py
├── RF_model.py
├── GDPa1_v1.2_sequences.csv
├── GDPa1_v1.2_20250814.csv
└── RF_top11_avg_feature_importance.png
```

## 2. File Description
- RF_utils.py:
Functions for converting raw antibody sequences into numerical features
(hydrophobicity, GRAVY, pI, charge, CDR-level features, etc.).
- RF_model.py:
End-to-end Random Forest pipeline:
feature generation
fold-based training and evaluation
performance metrics (R², RMSE, Spearman)
feature importance analysis
- GDPa1_v1.2_sequences.csv:
Input sequence data (VH/VL sequences, AHo-aligned sequences, fold labels).
-GDPa1_v1.2_20250814.csv:
Target property data containing HIC values.
- RF_top11_avg_feature_importance.png:
Plot of the top 11 most important features averaged across folds.


## 3. Input Data Format
- Sequence File: GDPa1_v1.2_sequences.csv
- Property File: GDPa1_v1.2_20250814.csv


## 4. Feature Overview
- Level 1: Global Fv (VH + VL combined)
- Level 2: Chain-level (VH and VL)
- Level 3: CDR-level (AHo-based)


## 5. How to Run
- Environment Setup
From the project root:
```
pip install -r requirements.txt
```
- Run the Random Forest Model
From the models/RF directory:
```
python RF_model.py
```

## 6. Training and Evaluation Strategy
- Cross-validation:
Uses predefined folds from hierarchical_cluster_IgG_isotype_stratified_fold, 
Each fold is used once as a held-out test set

- Model: 
RandomForestRegressor, 
n_estimators = 600, 
max_depth = 5, 
min_samples_leaf = 10

- Metrics: 
R² (train / test), 
RMSE (train / test), 
Spearman correlation on test folds

## 7. Feature Importance Analysis
- Feature importances are extracted from full models (Level 1 + 2 + 3)
- Importances are computed per fold
- Average rank and average importance are calculated across folds
- Top 11 features are visualized and saved as RF_top11_avg_feature_importance.png
