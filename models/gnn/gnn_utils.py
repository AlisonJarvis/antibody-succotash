import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.data import Dataset

# Get lists of pdb files and targets from dataframe
def get_pdb_and_targets(df, pdb_folder, target):

    antibody_ids = df['antibody_id']
    full_path_pdbs = [(pdb_folder + ab_id + '.pdb') for ab_id in antibody_ids]
    targets = list(df[target].to_numpy())
    return full_path_pdbs, targets

# Load train and test dataset using hierarchical cluster folds
def load_gnn_train_test(sequences_path, properties_path, emb_path, pdb_folder, target, fold):

    # Load sequences and properties, combine into relevant df
    sequences = pd.read_csv(sequences_path)
    properties = pd.read_csv(properties_path)
    embeddings = pd.read_pickle(emb_path)
    sequences_and_target = pd.merge(sequences[["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]], 
                                    properties[["antibody_id", target]], left_on="antibody_id", right_on="antibody_id")
    embeddings = embeddings.reset_index()
    embeddings.columns = ['antibody_id', 'embeddings']
    sequences_and_target = pd.merge(sequences_and_target, embeddings,  left_on="antibody_id", right_on="antibody_id")
    train_df = sequences_and_target[~(sequences_and_target['hierarchical_cluster_IgG_isotype_stratified_fold'] == fold)]
    test_df = sequences_and_target[sequences_and_target['hierarchical_cluster_IgG_isotype_stratified_fold'] == fold]

    # Drop nans
    train_df_clean = train_df.dropna()
    test_df_clean = test_df.dropna()

    # Get pdb files and targets for train and test
    train_pdbs, train_targets = get_pdb_and_targets(train_df_clean, pdb_folder, target)
    test_pdbs, test_targets = get_pdb_and_targets(test_df_clean, pdb_folder, target)

    return train_pdbs, train_df['embeddings'], train_targets, test_pdbs, test_df['embeddings'],  test_targets

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
    Loadd residues and c-alpha coordinates from a pdb file
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
def pairwise_features(residues, coords):
    """
    Compute pairwise features for a protein structure.
    residues: list of 3-letter amino acid codes
    coords: Nx3 array of alpha-carbon coordinates
    Returns:
        dist: NxN distance matrix
        hydro_diff: NxN hydrophobicity differences
        theta, phi, omega: NxN orientation angles
    """
    N = len(residues)
    
    # Map 3-letter to 1-letter with fallback X
    residues_1 = [three_to_one.get(r, "X") for r in residues]
    
    # Hydrophobicity, fallback maps to 0.0
    hydro = np.array([HYDRO.get(r, 0.0) for r in residues_1])
    
    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # Hydrophobicity difference
    hydro_diff = hydro[:, None] - hydro[None, :]
    
    # Orientation angles 
    vx, vy, vz = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
    theta = np.arctan2(vy, vx)
    phi   = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
    omega = np.arctan2(vz, vx)  # dihedral-like twist
    
    return dist, hydro_diff, theta, phi, omega

########## Graph from pdbs w/ cutoff ##########
def build_graph(pdb_path, residue_embeddings, target, cutoff=None):

    # Load residues and coordinates from pdb files
    residues, coords = load_pdb_residues(pdb_path)
    N = len(residues)

    # Node features - AA one hot and hydrophobicity
    x = np.zeros((N, 21), dtype=np.float32)
    for i, aa3 in enumerate(residues):
        aa1 = three_to_one.get(aa3, "X")
        idx = AA_TO_IDX.get(aa1, 20)
        x[i, idx] = 1
        x[i, 20] = HYDRO.get(aa1, 0.0)
    #x = np.hstack([x, residue_embeddings])
    print(x.shape[0] - residue_embeddings.shape[0])
    x = torch.tensor(x, dtype=torch.float32)

    # Builds NxN matrices of pairwise features
    dist, hydro_diff, theta, phi, omega = pairwise_features(residues, coords)

    # Build meshgrid of edge indices
    row, col = np.meshgrid(np.arange(N), np.arange(N))
    row = row.flatten()
    col = col.flatten()

    # Cutoff mask - true when the edge should be kept
    if cutoff is not None:
        mask = (dist.flatten() <= cutoff)
    else:
        mask = np.ones_like(dist.flatten(), dtype=bool)

    # Apply cutoff mask to edge indices
    edge_index = np.vstack([row[mask], col[mask]])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Flatten and stack edge attributes
    edge_attr_np = np.stack([
        dist.flatten(),
        hydro_diff.flatten(),
        theta.flatten(),
        phi.flatten(),
        omega.flatten()], axis=1)

    # Add the edge attributes with cutoff mask applied
    edge_attr = torch.tensor(edge_attr_np[mask], dtype=torch.float32)

    # Target value as torch.tensor
    y = torch.tensor([[target]], dtype=torch.float32)

    # Return as torch geometric Data object
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

###### Antibody Graph class ###########
class AntibodyGraphDataset(Dataset):
    def __init__(self, pdb_files, residue_embeddings, targets, cutoff=None):
        super().__init__()
        self.cutoff = cutoff

        # Precompute all of the graphs once
        self.graphs = []
        for pdb, embeddings, target in zip(pdb_files, residue_embeddings, targets):
            g = build_graph(pdb, embeddings, target, cutoff=self.cutoff)
            self.graphs.append(g)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]