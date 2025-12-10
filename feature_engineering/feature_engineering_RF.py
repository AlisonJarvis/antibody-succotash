import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from feature_utils_RF import create_features_from_raw_df
from train_test_division import divide_into_train_test

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import spearmanr


# Load in the csv file for sequences (input information)
csv_sequences = '../data/GDPa1_v1.2_sequences.csv'
sequences = pd.read_csv(csv_sequences)

# Load in the csv file for properties (output information)
csv_properties = '../data/GDPa1_v1.2_20250814.csv'
properties = pd.read_csv(csv_properties)

# Plot sequences lengths as a histogram

# Get sequence lengths
vh_sequence_lengths = [len(seq) for seq in sequences['vh_protein_sequence'].to_numpy()]
vl_sequence_lengths = [len(seq) for seq in sequences['vl_protein_sequence'].to_numpy()]

# Plot VH and VL sequence lengths
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(x=vh_sequence_lengths, stat='density', bins=10, color='red', label='VH')
sns.histplot(x=vl_sequence_lengths, stat='density', bins=10, color='blue', label='VL')
plt.legend()
ax.set_xlabel('Sequence Lengths')
ax.set_ylabel('Density')
plt.title('Distribution of Sequence Lengths for VH and VL Sequences')
fig.tight_layout()

# Now let's get the derived features from the sequences
sequence_features = create_features_from_raw_df(sequences)
sequence_features.head()

# features + fold + HIC -> merge
features_and_predictions = (
    sequence_features
    .merge(
        sequences[["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]],
        on="antibody_id",
        how="left",
    )
    .merge(
        properties[["antibody_id", "HIC"]],
        on="antibody_id",
        how="left",
    )
)
df = features_and_predictions.dropna(
    subset=["HIC", "hierarchical_cluster_IgG_isotype_stratified_fold"]
).reset_index(drop=True)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import spearmanr

# ============================================================
# I. Data Preparation
# ============================================================

# -------------------------------------------------
# I-1. Clean data: remove rows without HIC values
# -------------------------------------------------
df = features_and_predictions.copy()
df = df.dropna(subset=["HIC"]).reset_index(drop=True)

# -------------------------------------------------
# I-2. Define feature sets for each level
# -------------------------------------------------

# Level 1: Global Fv
level1_cols = [
    "fv_length",
    "fv_gravy",
    "fv_hydrophobic_count",
    "fv_pI",
    "fv_charge_pH7",
    "fv_frac_positive",
    "fv_frac_negative",
    "fv_frac_polar",
    "fv_frac_special",
]

# Level 2: VH/VL chain-level
#   - Physicochemical (sequence-derived)
#   - Composition & counts (sequence-derived)
#   - Categorical / annotation-based (subtype)

# 2-a. Physicochemical (sequence-derived, per chain)
level2_physchem_cols = [
    # length
    "vh_length", "vl_length",

    # GRAVY / hydrophobic
    "vh_gravy", "vl_gravy",
    "vh_hydrophobic_count", "vl_hydrophobic_count",

    # aromaticity / instability
    "vh_aromaticity", "vl_aromaticity",
    "vh_instability", "vl_instability",

    # pI / charge pH7
    "vh_pI", "vl_pI",
    "vh_charge_pH7", "vl_charge_pH7",

    # secondary structure fractions
    "vh_helix", "vh_turn", "vh_sheet",
    "vl_helix", "vl_turn", "vl_sheet",

    # MW / charge at other pH / extinction
    "vh_molecular_weight", "vl_molecular_weight",
    "vh_ph_7_35_charge", "vl_ph_7_35_charge",
    "vh_ph_7_45_charge", "vl_ph_7_45_charge",
    "vh_molar_extinction_reduced", "vh_molar_extinction_oxidized",
    "vl_molar_extinction_reduced", "vl_molar_extinction_oxidized",

    # AA class fractions
    "vh_frac_positive", "vh_frac_negative",
    "vh_frac_polar", "vh_frac_special",
    "vl_frac_positive", "vl_frac_negative",
    "vl_frac_polar", "vl_frac_special",

    # VH–VL relation (hydrophobicity)
    "vh_vl_hydrophobicity_gap",
    "vh_vl_hydrophobicity_ratio",
]

# 2-c. Composition & per-AA counts per chain (sequence-derived, Allison naming)
aa_count_cols = [
    c for c in df.columns
    if c.endswith("_vh_protein_sequence") or c.endswith("_vl_protein_sequence")
]

length_count_cols = [
    "vh_protein_sequence_length",
    "vl_protein_sequence_length",
]

derived_count_cols = [
    "vh_aromatic_count", "vl_aromatic_count",
    "vh_aliphatic_count", "vl_aliphatic_count",
]

# 2-e. Chain-level categorical / annotation-based (subtype one-hot)
subtype_cols = [
    c for c in df.columns
    if c.endswith("_hc_subtype") or c.endswith("_lc_subtype")
]

# extended Level 2 = Physicochemical + Composition/Counts + Categorical
level2_cols = (
    level2_physchem_cols
    + aa_count_cols
    + length_count_cols
    + derived_count_cols
    + subtype_cols
)

# Level 3: CDR-level (HCDR3 & LCDR1)
level3_cols = [
    # HCDR3 basic
    "HCDR3_length",
    "HCDR3_gravy",
    "HCDR3_hydrophobic_count",
    "HCDR3_aromaticity",
    "HCDR3_aromatic_cluster",

    # HCDR3 hydrophobicity-focused
    "HCDR3_hydrophobic_cluster_max_len",
    "HCDR3_hydrophobic_cluster_count",
    "HCDR3_hydrophobic_moment",
    "HCDR3_hydrophobic_density",
    "HCDR3_terminal_hydrophobicity",

    # LCDR1 basic
    "LCDR1_length",
    "LCDR1_gravy",
    "LCDR1_hydrophobic_count",
    "LCDR1_aromaticity",

    # LCDR1 hydrophobicity-focused
    "LCDR1_hydrophobic_cluster_max_len",
    "LCDR1_hydrophobic_cluster_count",
    "LCDR1_hydrophobic_moment",
    "LCDR1_hydrophobic_density",
    "LCDR1_terminal_hydrophobicity",
]

# -------------------------------------------------
# I-3. Validate feature columns (existing_cols)
# -------------------------------------------------

# Full model: Level 1 + 2 + 3
full_cols = level1_cols + level2_cols + level3_cols

# Ensure feature columns actually exist in the dataframe
def existing_cols(cols, df):
    return [c for c in cols if c in df.columns]

level1_cols = existing_cols(level1_cols, df)
level2_cols = existing_cols(level2_cols, df)
level3_cols = existing_cols(level3_cols, df)
full_cols   = existing_cols(full_cols,   df)

# ============================================================
# II. Modeling Utilities
# ============================================================

# -------------------------------------------------
# II-1. Fold-based train/test splitting
# -------------------------------------------------
def split_by_fold(
    df,
    target_col="HIC",
    fold_col="hierarchical_cluster_IgG_isotype_stratified_fold",
    fold=0,
):
    """
    Split the dataset into train/test according to a predefined fold column.

    - test  : rows where fold_col == fold
    - train : all remaining rows
    """

    if fold_col not in df.columns:
        raise ValueError(
            f"{fold_col} not found in dataframe. "
            f"Make sure the fold assignment column is included."
        )

    # boolean mask for test fold
    test_mask = df[fold_col] == fold

    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    return train_df, test_df


# -------------------------------------------------
# II-2. Random Forest training + R²/RMSE evaluation
# -------------------------------------------------
def train_rf_model_with_fold(
    train_df,
    test_df,
    feature_cols,
    target_col="HIC",
    model_name="",
    fold_id=None,
    n_estimators=600,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
):
    """
    Train and evaluate a Random Forest model using a predefined train/test split.
    No internal splitting is performed here.
    """

    feature_cols = existing_cols(feature_cols, train_df)

    if len(feature_cols) == 0:
        return None, {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": 0,
            "R2_train": np.nan,
            "R2_test": np.nan,
            "RMSE_train": np.nan,
            "RMSE_test": np.nan,
        }

    # Extract train and test matrices
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # Drop missing rows
    train_data = pd.concat([X_train, y_train], axis=1).dropna()
    test_data = pd.concat([X_test, y_test], axis=1).dropna()

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    if len(y_train) == 0 or len(y_test) == 0:
        return None, {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": len(feature_cols),
            "R2_train": np.nan,
            "R2_test": np.nan,
            "RMSE_train": np.nan,
            "RMSE_test": np.nan,
        }

    # Initialize Random Forest
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    # Train model
    rf.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results = {
        "model_name": model_name,
        "fold": fold_id,
        "n_samples_train": len(y_train),
        "n_samples_test": len(y_test),
        "n_features": len(feature_cols),
        "R2_train": r2_train,
        "R2_test": r2_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test,
    }

    return rf, results

# -------------------------------------------------
# II-3. Spearman correlation evaluation
# -------------------------------------------------

def compute_spearman_with_fold(
    rf,
    test_df,
    feature_cols,
    target_col="HIC",
    model_name="",
    fold_id=None,
):
    """
    Compute Spearman correlation for a given fold's test split.
    """

    if rf is None:
        return {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": 0,
            "Spearman_rho_test": np.nan,
            "p_value": np.nan,
        }

    feature_cols = existing_cols(feature_cols, test_df)

    if len(feature_cols) == 0:
        return {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": 0,
            "Spearman_rho_test": np.nan,
            "p_value": np.nan,
        }

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # Drop missing rows
    data = pd.concat([X_test, y_test], axis=1).dropna()
    X_test = data[feature_cols]
    y_test = data[target_col]

    if len(y_test) == 0:
        return {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": len(feature_cols),
            "Spearman_rho_test": np.nan,
            "p_value": np.nan,
        }

    y_pred_test = rf.predict(X_test)

    rho, pval = spearmanr(y_test, y_pred_test)

    return {
        "model_name": model_name,
        "fold": fold_id,
        "n_features": len(feature_cols),
        "Spearman_rho_test": rho,
        "p_value": pval,
    }
    

# ============================================================
# III. Cross-Fold Training & Evaluation
# ============================================================

# -------------------------------------------------
# III-1. Define folds and initialize result containers
# -------------------------------------------------
unique_folds = sorted(
    df["hierarchical_cluster_IgG_isotype_stratified_fold"]
      .dropna()
      .unique()
)

all_metrics = []      # stores RF metrics for all folds
all_spearman = []     # stores Spearman results for all folds
models_per_fold = {}  # optional: store RF models per fold


# -------------------------------------------------
# III-2. Train models for each feature level (Lvl1, Lvl2, Lvl3, Full)
# -------------------------------------------------

for fold_id in unique_folds:
    print(f"\n===== Fold {fold_id} =====")

    train_df_fold, test_df_fold = split_by_fold(
        df,
        target_col="HIC",
        fold_col="hierarchical_cluster_IgG_isotype_stratified_fold",
        fold=fold_id,
    )

    # ---- Level 1 ----
    rf_lvl1, res_lvl1 = train_rf_model_with_fold(
        train_df_fold, test_df_fold, level1_cols,
        target_col="HIC",
        model_name="Level 1 (Fv only)",
        fold_id=fold_id,
    )
    sp_lvl1 = compute_spearman_with_fold(
        rf_lvl1, test_df_fold, level1_cols,
        target_col="HIC",
        model_name="Level 1 (Fv only)",
        fold_id=fold_id,
    )

    # ---- Level 2 ----
    rf_lvl2, res_lvl2 = train_rf_model_with_fold(
        train_df_fold, test_df_fold, level2_cols,
        target_col="HIC",
        model_name="Level 2 (Chain-level)",
        fold_id=fold_id,
    )
    sp_lvl2 = compute_spearman_with_fold(
        rf_lvl2, test_df_fold, level2_cols,
        target_col="HIC",
        model_name="Level 2 (Chain-level)",
        fold_id=fold_id,
    )

    # ---- Level 3 ----
    rf_lvl3, res_lvl3 = train_rf_model_with_fold(
        train_df_fold, test_df_fold, level3_cols,
        target_col="HIC",
        model_name="Level 3 (CDR-level)",
        fold_id=fold_id,
    )
    sp_lvl3 = compute_spearman_with_fold(
        rf_lvl3, test_df_fold, level3_cols,
        target_col="HIC",
        model_name="Level 3 (CDR-level)",
        fold_id=fold_id,
    )

    # ---- Full ----
    rf_full, res_full = train_rf_model_with_fold(
        train_df_fold, test_df_fold, full_cols,
        target_col="HIC",
        model_name="Full (Level1+2+3)",
        fold_id=fold_id,
    )
    sp_full = compute_spearman_with_fold(
        rf_full, test_df_fold, full_cols,
        target_col="HIC",
        model_name="Full (Level1+2+3)",
        fold_id=fold_id,
    )

    models_per_fold[fold_id] = {
        "lvl1": rf_lvl1,
        "lvl2": rf_lvl2,
        "lvl3": rf_lvl3,
        "full": rf_full,
    }

    all_metrics.extend([res_lvl1, res_lvl2, res_lvl3, res_full])
    all_spearman.extend([sp_lvl1, sp_lvl2, sp_lvl3, sp_full])


metrics_df = pd.DataFrame(all_metrics)
spearman_df = pd.DataFrame(all_spearman)

print("\n=== Per-fold RF Metrics ===")
print(metrics_df)

print("\n=== Per-fold Spearman Correlations ===")
print(spearman_df)


# ============================================================
# IV. Summary Metrics Across Folds
# ============================================================
# ------------------------------------------------------------
# IV-1. Cross-validated R² and Spearman rho for all base models
# ------------------------------------------------------------

r2_summary = (
    metrics_df
    .groupby("model_name")["R2_test"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "CV_R2_mean", "std": "CV_R2_std"})
    .reset_index()
)

rho_summary = (
    spearman_df
    .groupby("model_name")["Spearman_rho_test"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "rho_mean", "std": "rho_std"})
    .reset_index()
)

summary_df = r2_summary.merge(rho_summary, on="model_name")

print("\n=== Cross-validated Summary Across Folds ===")
print(summary_df)


# ============================================================
# V. Feature Importance Analysis & Fold-Specific Top-N RF Modeling
# ============================================================

# ------------------------------------------------------------
# V-1. Extract per-fold feature importances from full RF models
# ------------------------------------------------------------

full_importances_list = []

for fold_id, models in models_per_fold.items():
    rf_full = models["full"]
    
    if rf_full is None:
        continue
    
    # Ensure feature alignment with DataFrame columns
    feature_cols = [c for c in full_cols if c in df.columns]
    
    # Extract feature importance values for this fold
    importances = rf_full.feature_importances_
    
    fold_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
        "fold": fold_id,
    })
    
    full_importances_list.append(fold_imp_df)

# Combine importance values across folds
full_importances = pd.concat(full_importances_list, axis=0)

print("\n=== Fold-Specific Feature Importances from Full Models (long format) ===")
print(full_importances.head())

# ------------------------------------------------------------
# V-2. Fold-specific Top-N feature RF models (fixed N)
# ------------------------------------------------------------

top_n = 11

all_metrics_top = []
all_spearman_top = []
models_per_fold_top = {}

for fold_id in unique_folds:
    print(f"\n===== Fold {fold_id} (Top {top_n} features, fold-specific) =====")

    # 1) Get importances for this fold only
    fold_imp = (
        full_importances[full_importances["fold"] == fold_id]
        .sort_values("importance", ascending=False)
    )

    # 2) Top-N features for this fold
    fold_top_features = fold_imp.head(top_n)["feature"].tolist()

    # 3) Split train/test for this fold
    train_df_fold, test_df_fold = split_by_fold(
        df,
        target_col="HIC",
        fold_col="hierarchical_cluster_IgG_isotype_stratified_fold",
        fold=fold_id,
    )

    # 4) Train RF using fold-specific Top-N features
    rf_top, res_top = train_rf_model_with_fold(
        train_df_fold,
        test_df_fold,
        fold_top_features,
        target_col="HIC",
        model_name=f"Full_top{top_n}_fold_specific",
        fold_id=fold_id,
    )

    # 5) Spearman evaluation
    sp_top = compute_spearman_with_fold(
        rf_top,
        test_df_fold,
        fold_top_features,
        target_col="HIC",
        model_name=f"Full_top{top_n}_fold_specific",
        fold_id=fold_id,
    )

    # 6) Save results
    models_per_fold_top[fold_id] = {
        "model": rf_top,
        "features": fold_top_features,
    }

    all_metrics_top.append(res_top)
    all_spearman_top.append(sp_top)


# ------------------------------------------------------------
# V-3. Per-fold Top-N results & CV summary (fixed N)
# ------------------------------------------------------------

metrics_top_df  = pd.DataFrame(all_metrics_top)
spearman_top_df = pd.DataFrame(all_spearman_top)

print("\n=== Per-fold RF Metrics (Top-N, fold-specific) ===")
print(metrics_top_df.to_string(index=False))

print("\n=== Per-fold Spearman (Top-N, fold-specific) ===")
print(spearman_top_df.to_string(index=False))


r2_top_summary = (
    metrics_top_df
    .groupby("model_name")["R2_test"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "CV_R2_mean", "std": "CV_R2_std"})
    .reset_index()
)

rho_top_summary = (
    spearman_top_df
    .groupby("model_name")["Spearman_rho_test"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "rho_mean", "std": "rho_std"})
    .reset_index()
)

top_summary_df = r2_top_summary.merge(rho_top_summary, on="model_name")

print("\n=== Cross-validated Summary (Top-N, fold-specific) ===")
print(top_summary_df.to_string(index=False))


# ------------------------------------------------------------
# (EXTRA) V-4. Top-N Optimization (1–25): global R2_mean, rho_mean, rho_std
# ------------------------------------------------------------

top_n_range = range(1, 26)

results_topn = []

for top_n in top_n_range:
    print(f"\n### Evaluating top_n = {top_n} ###")

    all_metrics_top = []
    all_spearman_top = []

    for fold_id in unique_folds:

        # 1) Fold-specific importances
        fold_imp = (
            full_importances[full_importances["fold"] == fold_id]
            .sort_values("importance", ascending=False)
        )

        # 2) Select top_n features
        fold_top_features = fold_imp.head(top_n)["feature"].tolist()

        # 3) Train/test split
        train_df_fold, test_df_fold = split_by_fold(
            df,
            target_col="HIC",
            fold_col="hierarchical_cluster_IgG_isotype_stratified_fold",
            fold=fold_id,
        )

        # 4) Train RF
        rf_top, res_top = train_rf_model_with_fold(
            train_df_fold,
            test_df_fold,
            fold_top_features,
            target_col="HIC",
            model_name=f"Full_top{top_n}_fold_specific",
            fold_id=fold_id,
        )

        # 5) Spearman evaluation
        sp_top = compute_spearman_with_fold(
            rf_top,
            test_df_fold,
            fold_top_features,
            target_col="HIC",
            model_name=f"Full_top{top_n}_fold_specific",
            fold_id=fold_id,
        )

        all_metrics_top.append(res_top)
        all_spearman_top.append(sp_top)


    # Convert to DataFrames for aggregation
    metrics_top_df  = pd.DataFrame(all_metrics_top)
    spearman_top_df = pd.DataFrame(all_spearman_top)

    # Aggregate across folds
    r2_mean  = metrics_top_df["R2_test"].mean()
    r2_std   = metrics_top_df["R2_test"].std()
    rho_mean = spearman_top_df["Spearman_rho_test"].mean()
    rho_std  = spearman_top_df["Spearman_rho_test"].std()

    # Save results
    results_topn.append({
        "top_n": top_n,
        "R2_mean": r2_mean,
        "R2_std":  r2_std,
        "rho_mean": rho_mean,
        "rho_std": rho_std
    })

# Convert full results to DF
results_topn_df = pd.DataFrame(results_topn)

print("\n=== Top-N Optimization Results ===")
print(results_topn_df)

# ------------------------------------------------------------
# (EXTRA) V-5. Visualization of Top-N Optimization (global)
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))
plt.plot(results_topn_df["top_n"], results_topn_df["R2_mean"], marker='o', label="R2_mean")
plt.plot(results_topn_df["top_n"], results_topn_df["R2_std"],  marker='o', label="R2_std")
plt.plot(results_topn_df["top_n"], results_topn_df["rho_mean"], marker='o', label="rho_mean")
plt.plot(results_topn_df["top_n"], results_topn_df["rho_std"],  marker='o', label="rho_std")

plt.xlabel("Top-N Features")
plt.ylabel("Metric Value")
plt.title("Top-N Feature Optimization (1–25)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
