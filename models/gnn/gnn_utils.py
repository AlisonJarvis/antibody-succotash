import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.data import Dataset
import torch
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool
)

import warnings
warnings.filterwarnings("ignore", "invalid value encountered in log", append=True)
warnings.filterwarnings("ignore", "divide by zero encountered in log", append=True)

# Get lists of pdb files and targets from dataframe
def get_pdb_and_targets(df, pdb_folder, target):

    antibody_ids = df['antibody_id']
    full_path_pdbs = [(pdb_folder + ab_id + '.pdb') for ab_id in antibody_ids]
    targets = list(df[target].to_numpy())
    return full_path_pdbs, targets

# Load train and test dataset using hierarchical cluster folds
def load_gnn_train_test(sequences_path, properties_path, pdb_folder, target, fold):

    # Load sequences and properties, combine into relevant df
    sequences = pd.read_csv(sequences_path)
    properties = pd.read_csv(properties_path)
    sequences_and_target = pd.merge(sequences[["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]], 
                                    properties[["antibody_id", target]], left_on="antibody_id", right_on="antibody_id")
    train_df = sequences_and_target[~(sequences_and_target['hierarchical_cluster_IgG_isotype_stratified_fold'] == fold)]
    test_df = sequences_and_target[sequences_and_target['hierarchical_cluster_IgG_isotype_stratified_fold'] == fold]

    # Drop nans
    train_df_clean = train_df.dropna()
    test_df_clean = test_df.dropna()

    # Get pdb files and targets for train and test
    train_pdbs, train_targets = get_pdb_and_targets(train_df_clean, pdb_folder, target)
    test_pdbs, test_targets = get_pdb_and_targets(test_df_clean, pdb_folder, target)

    return train_pdbs, train_targets, test_pdbs, test_targets

# Amino acid one hot encoding
AA_TO_IDX = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
    "X": 20  # fallback for unknown residues
}

# Kyte–Doolittle hydrophobicity scale with fallback
HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    "X": 0.0  # fallback hydrophobicity
}

# mapping of three-letter to one-letter codes
three_to_one = {k.upper(): v.upper() for k, v in IUPACData.protein_letters_3to1.items()}


######## PDB residue loading ###########
def load_pdb_residues(pdb_path):
    """
    Load residues and c-alpha coordinates from a pdb file
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    residues = []
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residues.append(residue.get_resname())
                    coords.append(residue["CA"].get_coord())

    coords = np.array(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coordinates malformed in {pdb_path}: {coords.shape}")
    return residues, coords


###### Calculates pairwise features #########
def pairwise_features(residues, coords, cutoff, log_dist=False):
    N = len(residues)

    residues_1 = [three_to_one.get(r, "X") for r in residues]
    hydro = np.array([HYDRO.get(r, 0.0) for r in residues_1])

    diff = coords[:,None,:] - coords[None,:,:]   # [N,N,3]
    dist = np.linalg.norm(diff, axis=-1)

    # Normalize distance
    max_d = cutoff if cutoff is not None else np.max(dist) + 1e-6
    dist_norm = dist / max_d
    inv_dist = 1.0 / (dist_norm + 1e-6)
    if log_dist: # If selected, use log_dist as feature
        t_dist = np.log(log_dist)
        max_d = np.log(cutoff) if cutoff is not None else np.max(t_dist) + 1e-6
        dist_norm = t_dist

    # Angles
    vx, vy, vz = diff[...,0], diff[...,1], diff[...,2]
    theta = np.arctan2(vy, vx)
    phi   = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
    omega = np.arctan2(vz, vx)

    # Sin/cos encode
    angle_features = np.stack([
        np.sin(theta), np.cos(theta),
        np.sin(phi),   np.cos(phi),
        np.sin(omega), np.cos(omega)
    ], axis=-1)

    hydro_diff = hydro[:,None] - hydro[None,:]

    return dist_norm, inv_dist, angle_features, hydro_diff

########## Graph from pdbs w/ cutoff ##########
def build_graph(pdb_path, target, cutoff=None, log_dist:bool=False):
    residues, coords = load_pdb_residues(pdb_path)
    N = len(residues)

    # Node features
    x = np.zeros((N, 21), dtype=np.float32)
    for i, aa3 in enumerate(residues):
        aa1 = three_to_one.get(aa3, "X")
        idx = AA_TO_IDX.get(aa1, 20)
        x[i, idx] = 1
        x[i, 20] = HYDRO.get(aa1, 0.0)
    x = torch.tensor(x, dtype=torch.float32)

    # Pairwise features
    dist_norm, inv_dist, angle_features, hydro_diff = pairwise_features(
        residues, coords, cutoff, log_dist
    )

    row, col = np.meshgrid(np.arange(N), np.arange(N))
    row = row.flatten()
    col = col.flatten()

    # Raw distances for cutoff
    dist_raw = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
    mask = dist_raw.flatten() <= cutoff if cutoff else np.ones(N*N, dtype=bool)

    edge_index = np.vstack([row[mask], col[mask]])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Build edge attributes
    edge_attr_np = np.concatenate([
        dist_norm.flatten()[:,None],
        inv_dist.flatten()[:,None],
        angle_features.reshape(-1,6),
        hydro_diff.flatten()[:,None]
    ], axis=1)

    edge_attr = torch.tensor(edge_attr_np[mask], dtype=torch.float32)

    y = torch.tensor([[target]], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

###### Antibody Graph class ###########
class AntibodyGraphDataset(Dataset):
    def __init__(self, pdb_files, targets, cutoff=None, log_dist:bool=False):
        super().__init__()
        self.cutoff = cutoff
        self.log_dist = log_dist

        # Precompute all of the graphs once
        self.graphs = []
        for pdb, target in zip(pdb_files, targets):
            g = build_graph(pdb, target, cutoff=self.cutoff, log_dist=self.log_dist)
            self.graphs.append(g)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
    
def global_mean_max_pool(x, batch,  size=None):
    mean_pool = global_mean_pool(x, batch, size)
    max_pool = global_max_pool(x, batch, size)
    return torch.cat([mean_pool, max_pool], dim=-1)


class ModelEvalTracker:
    def __init__(self):
        self.last_fold = 0
        self.last_iteration = 0
        self.evals = {}

    @property
    def cur_metric_avg(self):
        return np.mean(list(self.evals.values()))

    def update_metric(
        self,
        test_score: float,
        fold:int|None=None,
        iteration:int|None=None
    ):
        if fold is None and not iteration is None:
            raise ValueError("Iteration is changing but fold is not")

        # Update current fold/iteration if provided
        if fold: self.last_fold = fold
        if iteration: self.last_iteration = iteration
        
        self.evals[(iteration, fold)] = test_score