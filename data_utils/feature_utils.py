import pandas as pd
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def create_features_from_raw_df(input_df: pd.DataFrame) -> pd.DataFrame:
    '''Takes an input DataFrame of vh and vl antibody protein sequence strings + lc/hc subtypes and
    returns a new one with the following features:

    * BioPython computations of protein
        * aromaticity
        * isoelectric point
        * shape
        * instability
        * gravy
        * molecular weight
        * charge at pHs of 7.35 and 7.45 (the range of human blood)
        * molar extinction coefficients (both reduction and oxidized)
    * Counts of constituent amino acids
    * Protein lengths
    * One hot encodings of subtypes

    Args:
        input_dataframe (DataFrame): The input dataframe containing sequences and subtypes.
            Should be contained in the columns 'vh_protein_sequence', 'vl_protein_sequence', 'hc_subtype', and 'lc_subtype'.

    Returns:
        DataFrame: A dataframe with computed features.  
    '''
    # Initialize new dataframe with same index as input
    X = pd.DataFrame(data=[], index=input_df.index)

    # Iterate through component proteins of antibodies to get biopython features
    for component in ['vh', 'vl']:
        X[f'{component}_aromaticity'] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).aromaticity())
        X[f'{component}_pI'] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).isoelectric_point())
        X[[f'{component}_helix', f'{component}_turn', f'{component}_sheet']] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).secondary_structure_fraction()).apply(pd.Series)
        X[f'{component}_instability'] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).instability_index())
        X[f'{component}_gravy'] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).gravy())
        X[f'{component}_molecular_weight'] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).molecular_weight())
        X[f"{component}_ph_7_35_charge"] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).charge_at_pH(7.35))
        X[f"{component}_ph_7_45_charge"] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).charge_at_pH(7.45))
        X[[f"{component}_molar_extinction_reduced", f"{component}_molar_extinction_oxidized"]] = input_df[f'{component}_protein_sequence'].map(lambda prot: ProteinAnalysis(prot).molar_extinction_coefficient()).apply(pd.Series)

    # Iterate through sequences to add raw counts amino acid counts and protein lengths as features
    for col in [
        'vh_protein_sequence',
        'vl_protein_sequence'
    ]:
        
        # Add raw counts of each protein/base pair
        s = pd.json_normalize(input_df[col].apply(Counter))
        s = s.add_suffix(f"_{col}")
        X = X.merge(s, left_index=True, right_index=True, suffixes=("", ""))

        # Lastly add total length
        s_sum = s.sum(axis=1)
        s_sum.name = col + "_length"
        X = X.merge(s_sum, left_index = True, right_index = True, suffixes=("", ""))

    # Add one-hot encoded hc and lc subtypes
    X = X.merge(pd.get_dummies(input_df['hc_subtype']).add_suffix("_hc_subtype"),
        left_index=True, right_index=True, suffixes=("", "")         
    )
    X = X.merge(pd.get_dummies(input_df['lc_subtype']).add_suffix("_lc_subtype"),
        left_index=True, right_index=True, suffixes=("", "")         
    )

    X.fillna(0, inplace=True)

    return X
