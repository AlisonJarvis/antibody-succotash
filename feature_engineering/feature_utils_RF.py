import pandas as pd
import numpy as np
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---- Constants ----
HYDROPHOBIC_AA = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y']
AROMATIC_AA   = ['F', 'Y', 'W']
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
POLAR_AA    = set("STNQ")
SPECIAL_AA  = set("PGC")
HYDROPHOBIC_SET = set(HYDROPHOBIC_AA)
HYDRO_VALUES = {  # Kyte-Doolittle scale
    "A": 1.8, "I": 4.5, "L": 3.8, "M": 1.9, "V": 4.2,
    "F": 2.8, "W": -0.9, "Y": -1.3, "C": 2.5,
    "H": -3.2, "K": -3.9, "R": -4.5,
    "D": -3.5, "E": -3.5,
    "N": -3.5, "Q": -3.5,
    "G": -0.4, "P": -1.6, "S": -0.8, "T": -0.7
}

# AHo residue numbering ranges (inclusive)
HCDR3_RANGE_AHO = (108, 138)  # HCDR3: H108–H138
LCDR1_RANGE_AHO = (24, 42)    # LCDR1: L24–L42


# ---- Helper functions ----
def extract_cdr_from_aho(aligned_seq: str, start_pos: int, end_pos: int) -> str:
    """
    Extract a CDR sequence from an AHo-aligned sequence (length 149).
    Takes the window [start_pos, end_pos] (AHo numbering),
    removes gaps ('-'), and returns the contiguous amino-acid sequence.
    """
    if not isinstance(aligned_seq, str):
        return ""
    start_idx = start_pos - 1      # AHo numbering starts at 1
    end_idx = end_pos              # Python slicing end is exclusive
    window = aligned_seq[start_idx:end_idx]
    return "".join(ch for ch in window if ch.isalpha())


def cdr_basic_features(seq: str, prefix: str) -> dict:
    """
    Compute basic CDR features:
      - length
      - GRAVY hydrophobicity
      - hydrophobic residue count
      - aromaticity ratio (F/Y/W count divided by total length)
    Returns a dict where each key is prefixed with the provided prefix.
    """
    feats = {}
    if not seq:
        feats[f"{prefix}_length"] = 0
        feats[f"{prefix}_gravy"] = 0.0
        feats[f"{prefix}_hydrophobic_count"] = 0
        feats[f"{prefix}_aromaticity"] = 0.0
        return feats

    pa = ProteinAnalysis(seq)
    feats[f"{prefix}_length"] = len(seq)
    feats[f"{prefix}_gravy"] = pa.gravy()
    feats[f"{prefix}_hydrophobic_count"] = sum(seq.count(a) for a in HYDROPHOBIC_AA)

    aromatic_count = sum(seq.count(a) for a in AROMATIC_AA)
    feats[f"{prefix}_aromaticity"] = aromatic_count / len(seq)

    return feats


def has_aromatic_cluster(seq: str, min_run: int = 2) -> int:
    """
    Check whether the sequence has at least 'min_run' consecutive aromatic residues (F/Y/W).
    Returns:
      1 if an aromatic cluster exists (aggregation hotspot),
      0 otherwise.
    """
    if not seq:
        return 0

    run = 0
    for aa in seq:
        if aa in AROMATIC_AA:
            run += 1
            if run >= min_run:
                return 1
        else:
            run = 0
    return 0


def aa_class_fractions(seq: str) -> dict:
    """
    Return amino-acid class fractions for a sequence:
      - frac_positive: K/R/H
      - frac_negative: D/E
      - frac_polar:    S/T/N/Q
      - frac_special:  P/G/C
    """
    L = len(seq)
    if L == 0:
        return {
            "frac_positive": 0.0,
            "frac_negative": 0.0,
            "frac_polar": 0.0,
            "frac_special": 0.0,
        }

    return {
        "frac_positive": sum(seq.count(a) for a in POSITIVE_AA) / L,
        "frac_negative": sum(seq.count(a) for a in NEGATIVE_AA) / L,
        "frac_polar":    sum(seq.count(a) for a in POLAR_AA)    / L,
        "frac_special":  sum(seq.count(a) for a in SPECIAL_AA)  / L,
    }


def _chain_features(seq: str, ph: float = 7.0) -> dict:
    """
    Compute VH/VL chain-level features:
      - GRAVY hydrophobicity
      - hydrophobic residue count
      - aromaticity ratio
      - instability index (BioPython)
      - pI (isoelectric point)
      - net charge at given pH
      - amino-acid class fractions (positive/negative/polar/special)
    """
    if not seq:
        return {
            "gravy": 0.0,
            "hydrophobic_count": 0,
            "aromaticity": 0.0,
            "instability": 0.0,
            "pI": 0.0,
            "charge_pH7": 0.0,         # interpreted as pH ~7
            "helix": 0.0,
            "turn": 0.0,
            "sheet": 0.0,
            "molecular_weight": 0.0,
            "charge_pH7_35": 0.0,
            "charge_pH7_45": 0.0,
            "molar_extinction_reduced": 0.0,
            "molar_extinction_oxidized": 0.0,
            "frac_positive": 0.0,
            "frac_negative": 0.0,
            "frac_polar": 0.0,
            "frac_special": 0.0,
        }

    pa = ProteinAnalysis(seq)
    hydrophobic_count = sum(seq.count(a) for a in HYDROPHOBIC_AA)
    aromatic_count = sum(seq.count(a) for a in AROMATIC_AA)

    helix, turn, sheet = pa.secondary_structure_fraction()
    ext_red, ext_ox = pa.molar_extinction_coefficient()
    
    feats = {
        "gravy": pa.gravy(),
        "hydrophobic_count": hydrophobic_count,
        "aromaticity": aromatic_count / len(seq),
        "instability": pa.instability_index(),
        "pI": pa.isoelectric_point(),
        "charge_pH7": pa.charge_at_pH(ph),
        
        "helix": helix,
        "turn": turn,
        "sheet": sheet,
        "molecular_weight": pa.molecular_weight(),
        "charge_pH7_35": pa.charge_at_pH(7.35),
        "charge_pH7_45": pa.charge_at_pH(7.45),
        "molar_extinction_reduced": ext_red,
        "molar_extinction_oxidized": ext_ox
    } 
    feats.update(aa_class_fractions(seq))
    return feats

def max_hydrophobic_cluster_len(seq):
    """Return maximum length of consecutive hydrophobic residues."""
    max_len, cur = 0, 0
    for aa in seq:
        if aa in HYDROPHOBIC_SET:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len

def count_hydrophobic_clusters(seq):
    """Count number of hydrophobic clusters (consecutive stretches)."""
    count, cur = 0, 0
    for aa in seq:
        if aa in HYDROPHOBIC_SET:
            if cur == 0:
                count += 1
            cur += 1
        else:
            cur = 0
    return count

import math
def helix_hydrophobic_moment(seq):
    """Hydrophobic moment assuming alpha helix (100° per residue)."""
    moment_x, moment_y = 0.0, 0.0
    theta = math.radians(100)

    for i, aa in enumerate(seq):
        h = HYDRO_VALUES.get(aa, 0)
        angle = theta * i
        moment_x += h * math.cos(angle)
        moment_y += h * math.sin(angle)

    return math.sqrt(moment_x**2 + moment_y**2)

def terminal_hydrophobicity(seq, k=3):
    """Average hydrophobicity of N- and C-terminal segments."""
    if not seq or len(seq) < 2:
        return 0.0
    segment = seq[:k] + seq[-k:]
    vals = [HYDRO_VALUES.get(aa, 0) for aa in segment]
    return sum(vals) / len(vals)




# ---- Main feature builder ----
def create_features_from_raw_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build developability-related features from VH/VL sequences and aligned AHo sequences.

    Output includes:

    Level 1 (Fv global):
        - fv_length
        - fv_gravy
        - fv_hydrophobic_count
        - fv_pI
        - fv_charge_pH7
        - fv_frac_positive
        - fv_frac_negative
        - fv_frac_polar
        - fv_frac_special

    Level 2 (VH/VL chain-level and derived features):
        Physicochemical (sequence-derived):
            - vh_length, vl_length
            - vh_gravy, vl_gravy
            - vh_hydrophobic_count, vl_hydrophobic_count
            - vh_aromaticity, vl_aromaticity
            - vh_instability, vl_instability
            - vh_pI, vl_pI
            - vh_charge_pH7, vl_charge_pH7
            - vh_frac_positive / negative / polar / special
            - vl_frac_positive / negative / polar / special
            - vh_helix / turn / sheet, vl_helix / turn / sheet
            - vh_molecular_weight, vl_molecular_weight
            - vh_ph_7_35_charge / ph_7_45_charge
            - vl_ph_7_35_charge / ph_7_45_charge
            - vh_molar_extinction_(reduced/oxidized)
            - vl_molar_extinction_(reduced/oxidized)

        Composition & counts (sequence-derived):
            - per-AA counts per chain (e.g. A_vh_protein_sequence, ...)
            - vh_protein_sequence_length, vl_protein_sequence_length
            - vh_aromatic_count, vl_aromatic_count
            - vh_aliphatic_count, vl_aliphatic_count
            - vh_vl_hydrophobicity_gap
            - vh_vl_hydrophobicity_ratio

        Categorical / annotation-based (chain-level):
            - hc_subtype one-hot (e.g. IGHG1_hc_subtype, IGHM_hc_subtype, ...)
            - lc_subtype one-hot (e.g. IGK_lc_subtype, IGL_lc_subtype, ...)

    Level 3 (CDR level: HCDR3, LCDR1):
        - HCDR3_length, LCDR1_length
        - HCDR3_gravy, LCDR1_gravy
        - HCDR3_hydrophobic_count, LCDR1_hydrophobic_count
        - HCDR3_aromaticity
        - HCDR3_aromatic_cluster (≥2 consecutive aromatic residues)
    """

    X = pd.DataFrame(index=input_df.index)

    if "antibody_id" in input_df.columns:
        X["antibody_id"] = input_df["antibody_id"]

    # ------------------------
    # Level 1 — Global Fv
    # ------------------------
    vh_seqs = input_df["vh_protein_sequence"].fillna("").astype(str)
    vl_seqs = input_df["vl_protein_sequence"].fillna("").astype(str)
    fv_seqs = vh_seqs + vl_seqs

    # 1-a. Fv length
    X["fv_length"] = fv_seqs.str.len()

    # 1-b. Fv GRAVY
    X["fv_gravy"] = fv_seqs.map(
        lambda s: ProteinAnalysis(s).gravy() if s else 0.0
    )

    # 1-c. Hydrophobic residue count
    X["fv_hydrophobic_count"] = fv_seqs.map(
        lambda s: sum(s.count(aa) for aa in HYDROPHOBIC_AA)
    )

    # 1-d. Fv pI
    X["fv_pI"] = fv_seqs.map(
        lambda s: ProteinAnalysis(s).isoelectric_point() if s else 0.0
    )

    # 1-e. Fv net charge at pH 7
    X["fv_charge_pH7"] = fv_seqs.map(
        lambda s: ProteinAnalysis(s).charge_at_pH(7.0) if s else 0.0
    )

    # 1-f. Fv amino-acid class fractions
    fv_aa_fraction_dicts = fv_seqs.map(aa_class_fractions)
    fv_frac_df = pd.DataFrame(list(fv_aa_fraction_dicts), index=input_df.index)
    for col in fv_frac_df.columns:
        X[f"fv_{col}"] = fv_frac_df[col]

    # ------------------------
    # Level 2 — VH / VL chain-level and derived sequence features
    # ------------------------

    # 2-a. Physicochemical (sequence-derived, per chain)
    for chain in ["vh", "vl"]:
        seq_col = f"{chain}_protein_sequence"
        seqs = input_df[seq_col].fillna("").astype(str)

        # length
        X[f"{chain}_length"] = seqs.str.len()

        # GRAVY / hydrophobic_count / aromaticity / instability / pI / charge / aa fractions
        chain_feature_dicts = seqs.map(lambda s: _chain_features(s, ph=7.0))
        chain_df = pd.DataFrame(list(chain_feature_dicts), index=input_df.index)

        X[f"{chain}_gravy"] = chain_df["gravy"]
        X[f"{chain}_hydrophobic_count"] = chain_df["hydrophobic_count"]
        X[f"{chain}_aromaticity"] = chain_df["aromaticity"]
        X[f"{chain}_instability"] = chain_df["instability"]
        X[f"{chain}_pI"] = chain_df["pI"]
        X[f"{chain}_charge_pH7"] = chain_df["charge_pH7"]

        # secondary structure / MW / charge at different pH / extinction (Allison-style)
        X[f"{chain}_helix"] = chain_df["helix"]
        X[f"{chain}_turn"] = chain_df["turn"]
        X[f"{chain}_sheet"] = chain_df["sheet"]
        X[f"{chain}_molecular_weight"] = chain_df["molecular_weight"]
        X[f"{chain}_ph_7_35_charge"] = chain_df["charge_pH7_35"]
        X[f"{chain}_ph_7_45_charge"] = chain_df["charge_pH7_45"]
        X[f"{chain}_molar_extinction_reduced"] = chain_df["molar_extinction_reduced"]
        X[f"{chain}_molar_extinction_oxidized"] = chain_df["molar_extinction_oxidized"]

        # amino-acid class fractions
        X[f"{chain}_frac_positive"] = chain_df["frac_positive"]
        X[f"{chain}_frac_negative"] = chain_df["frac_negative"]
        X[f"{chain}_frac_polar"] = chain_df["frac_polar"]
        X[f"{chain}_frac_special"] = chain_df["frac_special"]

    # 2-b. VH–VL derived hydrophobicity features (sequence-derived, Fv-level but from chains)
    X["vh_vl_hydrophobicity_gap"] = X["vh_gravy"] - X["vl_gravy"]
    X["vh_vl_hydrophobicity_ratio"] = X["vh_hydrophobic_count"] / (
        X["vl_hydrophobic_count"] + 1e-6
    )

    # 2-c. Composition & per-AA counts per chain (sequence-derived, Allison naming)
    for col in ["vh_protein_sequence", "vl_protein_sequence"]:
        if col in input_df.columns:
            seq_series = input_df[col].fillna("").astype(str)

            # per-AA raw counts: e.g. A_vh_protein_sequence, C_vh_protein_sequence, ...
            aa_counts = seq_series.map(Counter)
            aa_df = pd.json_normalize(aa_counts)
            aa_df = aa_df.add_suffix(f"_{col}")
            aa_df.index = input_df.index  # align indices

            for c in aa_df.columns:
                X[c] = aa_df[c]

            # total length (Allison naming)
            length_col = f"{col}_length"
            X[length_col] = aa_df.sum(axis=1)

    # 2-d. Derived aromatic / aliphatic counts per chain (sequence-derived, Allison-style)
    for chain in ["vh", "vl"]:
        base_col = f"{chain}_protein_sequence"

        # aromatic count: F/Y/W
        aromatic_indices = ["F", "Y", "W"]
        aromatic_cols = [
            f"{aa}_{base_col}" for aa in aromatic_indices
            if f"{aa}_{base_col}" in X.columns
        ]
        if aromatic_cols:
            X[f"{chain}_aromatic_count"] = X[aromatic_cols].sum(axis=1)
        else:
            X[f"{chain}_aromatic_count"] = 0

        # aliphatic count: A/V/I/L
        aliphatic_indices = ["A", "V", "I", "L"]
        aliphatic_cols = [
            f"{aa}_{base_col}" for aa in aliphatic_indices
            if f"{aa}_{base_col}" in X.columns
        ]
        if aliphatic_cols:
            X[f"{chain}_aliphatic_count"] = X[aliphatic_cols].sum(axis=1)
        else:
            X[f"{chain}_aliphatic_count"] = 0

    # 2-e. Chain-level categorical / annotation-based features (subtype one-hot)
    if "hc_subtype" in input_df.columns:
        hc_dummies = pd.get_dummies(input_df["hc_subtype"]).add_suffix("_hc_subtype")
        for c in hc_dummies.columns:
            X[c] = hc_dummies[c]

    if "lc_subtype" in input_df.columns:
        lc_dummies = pd.get_dummies(input_df["lc_subtype"]).add_suffix("_lc_subtype")
        for c in lc_dummies.columns:
            X[c] = lc_dummies[c]


    # ------------------------
    # Level 3 — CDR (HCDR3, LCDR1)
    # ------------------------
    has_heavy_aho = "heavy_aligned_aho" in input_df.columns
    has_light_aho = "light_aligned_aho" in input_df.columns

    # Initialize columns
    cdr_cols = [
        # HCDR3 basic
        "HCDR3_length",
        "HCDR3_gravy",
        "HCDR3_hydrophobic_count",
        "HCDR3_aromaticity",
        "HCDR3_aromatic_cluster",

        # HCDR3 hydrophobicity-related
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

        # LCDR1 hydrophobicity-related
        "LCDR1_hydrophobic_cluster_max_len",
        "LCDR1_hydrophobic_cluster_count",
        "LCDR1_hydrophobic_moment",
        "LCDR1_hydrophobic_density",
        "LCDR1_terminal_hydrophobicity",
    ]
    # Initialize all CDR feature columns
    for col in cdr_cols:
        X[col] = 0.0

    # ------------------------------------
    # 3-a. HCDR3
    # ------------------------------------
    if has_heavy_aho:
        h_start, h_end = HCDR3_RANGE_AHO
        hcdr3_seqs = input_df["heavy_aligned_aho"].map(
            lambda s: extract_cdr_from_aho(s, h_start, h_end)
        )

        # Basic features
        hcdr3_feature_dicts = hcdr3_seqs.map(
            lambda seq: cdr_basic_features(seq, prefix="HCDR3")
        )
        hcdr3_df = pd.DataFrame(list(hcdr3_feature_dicts), index=input_df.index)

        for col in ["HCDR3_length", "HCDR3_gravy",
                    "HCDR3_hydrophobic_count", "HCDR3_aromaticity"]:
            if col in hcdr3_df.columns:
                X[col] = hcdr3_df[col]

        X["HCDR3_aromatic_cluster"] = hcdr3_seqs.map(has_aromatic_cluster)

        # ------------------------------------
        # NEW: Additional Hydrophobicity Features
        # ------------------------------------
        X["HCDR3_hydrophobic_cluster_max_len"] = hcdr3_seqs.map(max_hydrophobic_cluster_len)
        X["HCDR3_hydrophobic_cluster_count"] = hcdr3_seqs.map(count_hydrophobic_clusters)
        X["HCDR3_hydrophobic_moment"] = hcdr3_seqs.map(helix_hydrophobic_moment)
        X["HCDR3_hydrophobic_density"] = (
            X["HCDR3_hydrophobic_count"] / X["HCDR3_length"].replace(0, np.nan)
        )
        X["HCDR3_terminal_hydrophobicity"] = hcdr3_seqs.map(terminal_hydrophobicity)


    # ------------------------------------
    # 3-b. LCDR1
    # ------------------------------------
    if has_light_aho:
        l_start, l_end = LCDR1_RANGE_AHO
        lcdr1_seqs = input_df["light_aligned_aho"].map(
            lambda s: extract_cdr_from_aho(s, l_start, l_end)
        )

        lcdr1_feature_dicts = lcdr1_seqs.map(
            lambda seq: cdr_basic_features(seq, prefix="LCDR1")
        )
        lcdr1_df = pd.DataFrame(list(lcdr1_feature_dicts), index=input_df.index)

        for col in ["LCDR1_length", "LCDR1_gravy",
                    "LCDR1_hydrophobic_count", "LCDR1_aromaticity"]:
            if col in lcdr1_df.columns:
                X[col] = lcdr1_df[col]

        # ------------------------------------
        # NEW: Additional Hydrophobicity Features
        # ------------------------------------
        X["LCDR1_hydrophobic_cluster_max_len"] = lcdr1_seqs.map(max_hydrophobic_cluster_len)
        X["LCDR1_hydrophobic_cluster_count"] = lcdr1_seqs.map(count_hydrophobic_clusters)
        X["LCDR1_hydrophobic_moment"] = lcdr1_seqs.map(helix_hydrophobic_moment)
        X["LCDR1_hydrophobic_density"] = (
            X["LCDR1_hydrophobic_count"] / X["LCDR1_length"].replace(0, np.nan)
        )
        X["LCDR1_terminal_hydrophobicity"] = lcdr1_seqs.map(terminal_hydrophobicity)


    X = X.fillna(0)

    return X
