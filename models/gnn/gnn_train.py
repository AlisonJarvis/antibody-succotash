import wandb
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr
import yaml
from gnn_models import FlexibleGNN
from gnn_utils import load_gnn_train_test, AntibodyGraphDataset

######## Spearman correlation on torch tensor #########
def spearman_corr(y_true, y_pred):
    """"
    Util: computes spearman rho for torch tensors
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return spearmanr(y_true, y_pred)[0]

############# Training Loop ##################
def train_one_epoch(model, loader, optimizer, device, loss_type):
    model.train()
    total_loss = 0
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        if loss_type == 'mse':
            loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
        elif loss_type == 'nll':
            loss = F.nll_loss(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds.append(out.detach().cpu())
        trues.append(batch.y.cpu())

    preds = torch.cat(preds).squeeze()
    trues = torch.cat(trues).squeeze()

    avg_loss = total_loss / len(loader.dataset)
    rho = spearman_corr(trues, preds)

    return avg_loss, rho

################ Validation Loop ######################
@torch.no_grad()
def evaluate(model, loader, device, loss_type):
    model.eval()
    total_loss = 0
    preds = []
    trues = []

    # Iterate through each batch
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        # Define loss type
        if loss_type == 'mse':
            loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
        elif loss_type == 'nll':
            loss = F.nll_loss(out.squeeze(), batch.y.squeeze())

        total_loss += loss.item() * batch.num_graphs
        preds.append(out.detach().cpu())
        trues.append(batch.y.cpu())

    preds = torch.cat(preds).squeeze()
    trues = torch.cat(trues).squeeze()

    avg_loss = total_loss / len(loader.dataset)
    rho = spearman_corr(trues, preds)

    return avg_loss, rho

########### Full Cross Validation Loop #################
def train_cross_validation(config, sequences_path, properties_path, pdb_folder, target):

    # Define the device (mps or cpu)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    all_fold_results = []

    # Initialize wandb for this fold
    wandb.init(project=config["project_name"],
                config=config)

    # Cross validation - for folds 0–4
    for fold in range(5):

        print(f"\nFold {fold}\n")

        train_pdbs, train_targets, test_pdbs, test_targets = load_gnn_train_test(sequences_path, properties_path, pdb_folder, target, fold)

        train_dataset = AntibodyGraphDataset(train_pdbs, train_targets, cutoff=config["cutoff"])
        test_dataset = AntibodyGraphDataset(test_pdbs, test_targets, cutoff=config["cutoff"])

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False)

        # Create model for this fold
        config["input_dim"] = train_dataset[0].x.shape[1]
        model = FlexibleGNN(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        best_test_rho = -999
        best_model_state = None

        # Training loop
        for epoch in range(1, config["epochs"] + 1):

            train_loss, train_rho = train_one_epoch(model, train_loader, optimizer, device, config["criterion"])
            test_loss, test_rho = evaluate(model, test_loader, device, config["criterion"])

            # Log to wandb
            wandb.log({
                f"fold_{fold}/epoch": epoch,
                f"fold_{fold}/train_loss": train_loss,
                f"fold_{fold}/test_loss": test_loss,
                f"fold_{fold}/train_spearman": train_rho,
                f"fold_{fold}/test_spearman": test_rho
            })

            print(f"Epoch {epoch} | "
                  f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f} | "
                  f"train_rho={train_rho:.3f}, test_rho={test_rho:.3f}")

            # Select best model by highest test spearman rho
            if test_rho > best_test_rho:
                best_test_rho = test_rho
                best_model_state = model.state_dict()

        # Out of epoch loop - get best test spearman rho as eval metric
        wandb.summary["best_test_spearman"] = best_test_rho
        all_fold_results.append(best_test_rho)

        # Save best model for this fold
        torch.save(best_model_state, f"best_model_fold{fold}.pt")

    wandb.finish()

    # After all folds run - print all fold results
    print("\nCross-validation complete.")
    print("Fold Spearman rho:", all_fold_results)
    print("Mean rho:", sum(all_fold_results) / len(all_fold_results))

###### Main Run #######
if __name__ == '__main__':
    # Load in config
    config_path = './gnn_config.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Data loading parameters
    sequences_path = './GDPa1_v1.2_sequences.csv'
    properties_path = './GDPa1_v1.2_20250814.csv'
    pdb_folder = './pdb_files/'
    target = 'HIC'

    # Run cross-validation
    train_cross_validation(config, sequences_path, properties_path, pdb_folder, target)

