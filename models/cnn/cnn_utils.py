import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def get_embeddings_list_channels(X, pca_model, padded_length:int=250):
    def pad(input):
        # Remove what are essentially bookend values
        input = input[1:-2]
        pad_number = padded_length - input.shape[0]
        return np.pad(input, ((pad_number, 0), (0, 0))).T.reshape(input.shape[1], padded_length)
    return np.stack(list(X['embeddings'].apply(pca_model.transform).apply(pad)))

def get_channels_embedding_data_for_fold(X, test_fold:int, pca_model: PCA, fold_key:str):
    fold_sr = X[fold_key]
    test_idxs = (fold_sr == test_fold)
    X_test = X.loc[test_idxs]

    test_data = TensorDict({
            "embedding_data": torch.Tensor(get_embeddings_list_channels(X_test, pca_model)),
            "subtypes": torch.Tensor(X_test[['IgG2_hc_subtype', 'IgG4_hc_subtype', 'Lambda_lc_subtype']].to_numpy()),
            "gravy": torch.Tensor(X_test[['vh_gravy', 'vl_gravy']].to_numpy()),
            "target": torch.Tensor(list(X_test['target'])),
            "fold": torch.Tensor(X_test[fold_key].to_numpy()),
        },
        batch_size = X_test.shape[0]
    )
    return test_data

def get_test_fold_dataloaders(
    test_fold: int,
    X: pd.DataFrame,
    batch_size: int,
    ndims: int=16,
    fold_key: str='hierarchical_cluster_IgG_isotype_stratified_fold'
)-> tuple[Dataset, Dataset, DataLoader, DataLoader, PCA]:
    
    fold_sr = X[fold_key]
    test_idxs = (fold_sr == test_fold)
    train_idxs = (fold_sr != test_fold)
    pca = PCA(ndims)
    pca.fit(np.concatenate(list(X['embeddings'].loc[train_idxs])))

    fold_data = []
    for i in range(5):
        fold_data.append(get_channels_embedding_data_for_fold(X, i, pca, fold_key))

    test_data = fold_data[test_fold]
    train_data = torch.cat(tuple(fold for i, fold in enumerate(fold_data) if i != test_fold), dim=0)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=lambda x: x)
    test_loader =  DataLoader(test_data, batch_size=38, collate_fn=lambda x: x)
    return test_data, train_data, test_loader, train_loader, pca

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


def generate_features(property:str = "HIC") -> pd.DataFrame:
    fp = Path(__file__)
    contest_data_path = fp.parent.parent.parent.joinpath('data', 'GDPa1_v1.2_20250814.csv').absolute().as_posix()
    contest_data = pd.read_csv(contest_data_path)
    non_embedding_features_df = create_features_from_raw_df(contest_data)
    non_embedding_features_df.set_index(contest_data['antibody_id'], inplace=True)

    embedding_data_path = fp.parent.parent.joinpath('gnn', 'embeddings_sr.pkl').absolute().as_posix()
    embedding_df = pd.DataFrame()
    embedding_df['embeddings'] = pd.read_pickle(embedding_data_path)
    embedding_df=embedding_df.reset_index()
    embedding_df['target'] = contest_data[property]
    embedding_df['hierarchical_cluster_IgG_isotype_stratified_fold'] = contest_data['hierarchical_cluster_IgG_isotype_stratified_fold']
    embedding_df=embedding_df.set_index('antibody_id')
    X = embedding_df[['embeddings', 'target', 'hierarchical_cluster_IgG_isotype_stratified_fold']].merge(non_embedding_features_df, left_index=True, right_index=True)
    X = X.dropna()
    return X