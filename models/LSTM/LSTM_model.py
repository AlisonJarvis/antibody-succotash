import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from LSTM_utils import create_features_from_raw_df
from LSTM_utils import get_model_embeddings

csv_sequences = "data/GDPa1_v1.2_sequences.csv" 
csv_properties = "data/GDPa1_v1.2_20250814.csv"

### PARAMETERS ###
Y = "HIC"
batch_size = 64
epochs = 10
lr = 1e-3
random_seed = 42


### Torch Setup ###
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(random_seed)
np.random.seed(random_seed)


seq_df = pd.read_csv(csv_sequences)
prop_df = pd.read_csv(csv_properties)

target_df = prop_df[["antibody_id", Y]]

seq_target = seq_df.merge(target_df, on="antibody_id", how="inner")
sequence_features = create_features_from_raw_df(seq_target)

# concat target and features
df = pd.concat(
    [seq_target.reset_index(drop=True),
     sequence_features.drop(columns=["antibody_id"]).reset_index(drop=True)],
    axis=1)

# transformer embeddings
df["seq_emb"] = get_model_embeddings(df)

engineered_cols = [
    "vh_hydrophobic_count",
    "Y_vh_protein_sequence",
    "vh_aromatic_count",
    "vh_molar_extinction_oxidized",
    "vh_molar_extinction_reduced",
    "vh_aromaticity",
    "vh_protein_sequence_length",
    "vl_hydrophobic_count",
    "vh_pI",
    "vh_ph_7_35_charge",
    "vh_ph_7_45_charge"
]

df = df.dropna(subset=[Y]).reset_index(drop=True)

class AntibodySeqDataset(Dataset):
    def __init__(self, df, feature_cols=None):
        self.seq_embs = df["seq_emb"].to_list()
        self.y = df[Y].astype(float).values
        if feature_cols:
            self.features = df[feature_cols].astype(np.float32).values
        else:
            self.features = None

    def __len__(self):
        return len(self.y)

    # what fed into model
    def __getitem__(self, idx):
        seq = torch.tensor(self.seq_embs[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)

        sample = {"seq": seq, "target": target}

        if self.features is not None:
            sample["features"] = torch.tensor(self.features[idx], dtype=torch.float32)

        return sample
    

def collate_fn(batch):
    seqs = [b["seq"] for b in batch]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)

    # pad
    padded = nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=0.0
    )

    batch_out = {
        "seq": padded,
        "seq_lengths": lengths,
        "target": torch.stack([b["target"] for b in batch]),
    }

    if "features" in batch[0]:
        batch_out["features"] = torch.stack([b["features"] for b in batch])

    return batch_out

class AntibodyLSTMModel(nn.Module):
    def __init__(self,
                 input_size=20,
                 hidden_size=64,
                 num_layers=1,
                 dropout=0.1,
                 engineered_feat_dim=0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        input_dim = hidden_size + engineered_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq, seq_lengths, features=None):

        packed = nn.utils.rnn.pack_padded_sequence(
            seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output, (hidden, cell) = self.lstm(packed)
        seq_repr = hidden[-1] 

        combined = seq_repr
        if features is not None:
            combined = torch.cat([combined, features], dim=1)

        out = self.mlp(combined).squeeze(-1)
        return out

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        seq = batch["seq"].to(device)
        seq_lengths = batch["seq_lengths"].to(device)
        targets = batch["target"].to(device)

        feats = batch.get("features")
        if feats is not None:
            feats = feats.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(seq, seq_lengths, features=feats)
            loss = criterion(preds, targets)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if np.std(all_preds) < 1e-8 or np.std(all_targets) < 1e-8:
        spearman = np.nan
    else:
        rho, _ = spearmanr(all_targets, all_preds)
        spearman = float(rho)

    return avg_loss, spearman


for test_fold in range(5):

    
    train_df, test_df = df.loc[df['hierarchical_cluster_IgG_isotype_stratified_fold']!=test_fold], df.loc[df['hierarchical_cluster_IgG_isotype_stratified_fold']==test_fold]

    train_df.loc[:, engineered_cols] = train_df[engineered_cols].astype(np.float64)
    test_df.loc[:, engineered_cols] = test_df[engineered_cols].astype(np.float64)

    # normalize all the other features
    scaler = StandardScaler()

    # implemented these warning catches because pandas was being really picky
    # even though I assigned float64 to literally everything
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*incompatible dtype.*")
        train_df.loc[:, engineered_cols] = scaler.fit_transform(train_df[engineered_cols])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*incompatible dtype.*")
        test_df.loc[:, engineered_cols]  = scaler.transform(test_df[engineered_cols])

    train_dataset = AntibodySeqDataset(train_df, feature_cols=engineered_cols)
    test_dataset  = AntibodySeqDataset(test_df,  feature_cols=engineered_cols)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    emb_dim = train_df["seq_emb"].iloc[0].shape[1]

    model = AntibodyLSTMModel(
        input_size=emb_dim,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        engineered_feat_dim=len(engineered_cols)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_rho = run_epoch(train_loader, train=True)
        val_loss, val_rho = run_epoch(test_loader, train=False)
        print(
            f"Epoch {epoch:02d} : "
            f"train ρ: {train_rho:.3f} | "
            f"val ρ: {val_rho:.3f}"
        )
    print(f"Finished running Fold #{test_fold}")