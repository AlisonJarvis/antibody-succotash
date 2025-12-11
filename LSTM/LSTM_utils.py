import pandas as pd
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

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
    X["antibody_id"] = input_df["antibody_id"]

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

    # Derived sequence features
    for col in ['vh', 'vl']:
        hydrophobic_indices = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y']
        hydrophobic_keys = [indc + '_' + col + '_protein_sequence' for indc in hydrophobic_indices]
        hydrophobic_sum = X[hydrophobic_keys].sum(axis=1)
        h_key = col + '_hydrophobic_count'
        X[h_key] = hydrophobic_sum

        aromatic_indices = ['F', 'Y', 'W']
        aromatic_keys = [indc + '_' + col + '_protein_sequence' for indc in aromatic_indices]
        aromatic_sum = X[aromatic_keys].sum(axis=1)
        ar_key = col + '_aromatic_count'
        X[ar_key] = aromatic_sum
        
        aliphatic_indices = ['A', 'V', 'I', 'L']
        alphatic_keys = [indc + '_' + col + '_protein_sequence' for indc in aliphatic_indices]
        aliphatic_sum = X[alphatic_keys].sum(axis=1)
        al_key = col + '_aliphatic_count'
        X[al_key] = aliphatic_sum

    # Add one-hot encoded hc and lc subtypes
    X = X.merge(pd.get_dummies(input_df['hc_subtype']).add_suffix("_hc_subtype"),
        left_index=True, right_index=True, suffixes=("", "")         
    )
    X = X.merge(pd.get_dummies(input_df['lc_subtype']).add_suffix("_lc_subtype"),
        left_index=True, right_index=True, suffixes=("", "")         
    )

    X.fillna(0, inplace=True)

    return X


def get_model_embeddings(sequence_df: pd.DataFrame) -> pd.Series:
    '''
    Given an input `DataFrame` with `"vh_protein_sequence"` and `"vl_protein_sequence"` columns,
    generates an embedding for each token using the p-IgGen transformer model. Returns results as
    a list of numpy arrays.

    Args:
        sequence_df (DataFrame): The input dataframe containing the vh and vl protein sequences.

    Returns:
        Series: A series with elements consisting of token level embeddings of each sequence.
    '''
    model_name = "ollieturnbull/p-IgGen"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Paired sequence handling: Concatenate heavy and light chains and add beginning ("1") and end ("2") tokens
    # (e.g. ["EVQLV...", "DIQMT..."] -> "1E V Q L V ... D I Q M T ... 2")
    sequences = [
        "1" + " ".join(heavy) + " ".join(light) + "2"
        for heavy, light in zip(
            sequence_df["vh_protein_sequence"],
            sequence_df["vl_protein_sequence"],
        )
    ]

    # Load model
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    batch_size = 16
    full_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = tokenizer(sequences[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
        outputs = model(batch["input_ids"].to(device), return_rep_layers=[-1], output_hidden_states=True)
        embeddings = outputs["hidden_states"][-1].detach().cpu().numpy()
        for i in range(embeddings.shape[0]):
            full_embeddings.append(embeddings[i])
    return pd.Series(full_embeddings, index=sequence_df.index)