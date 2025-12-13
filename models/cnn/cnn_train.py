from dataclasses import dataclass
import traceback
import pandas as pd
import numpy as np
import scipy as sp
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import wandb
from pprint import pprint, pformat
import logging
import traceback

from cnn_utils import get_test_fold_dataloaders, generate_features
from cnn_models import ConvModelFullVector

logger = logging.Logger(__name__)

file_handler = logging.FileHandler('cnn_train.txt', mode='a')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@dataclass
class TrainContext:
    device:str='cpu'
    batch_size:int=32
    epoch:int=0


def train_loop(dataloader, model, loss_fn, optimizer, ctx: TrainContext):
    # Set the model to training mode - important for batch normalization and dropout layers
    device = ctx.device
    model.to(device)
    model.train()
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        X = data["embedding_data"].to(device)
        y = data["target"].to(device)
        subtypes = data['subtypes'].to(device)
        gravy = data['gravy'].to(device)
    
        optimizer.zero_grad()

        try:
            pred = model(X, subtypes, gravy)
            pred = torch.reshape(pred, (-1, 2))
            loss = loss_fn(y, pred[:, 0], pred[:, 1])
        except Exception as e:
            data_printout = pformat(data, indent=4)
            logger.log(level=logging.DEBUG, msg=data_printout, exc_info=True)
            raise e

        # Backpropagation
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test_loop(dataloader, model, loss_fn, train_data, ctx: TrainContext):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = ctx.batch_size
    device = ctx.device
    test_loss, correct, spearman = 0, 0, []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    for data in dataloader:
        model.eval()
        
        X = data["embedding_data"].to(device)
        y = data["target"].to(device)
        subtypes = data['subtypes'].to(device)
        gravy = data['gravy'].to(device)
        pred = model(X, subtypes, gravy)
        test_loss += loss_fn(y, pred[:, 0], pred[:, 1])
        try:
            spearman.append(sp.stats.spearmanr(y.cpu().detach().numpy(), pred.cpu().detach().numpy()[:, 0]).statistic)
        except Exception:
            spearman.append(-1)
    
    y_pred_train = (
        torch.squeeze(
            model(train_data["embedding_data"].to(device), train_data["subtypes"].to(device), train_data["gravy"].to(device))[:, 0],
        )
        .cpu()
        .detach()
        .numpy()
    )
    y_train = torch.squeeze(train_data["target"]).detach().numpy()
    train_spearman = sp.stats.spearmanr(y_train, y_pred_train).statistic
    test_loss /= num_batches
    correct /= size
    spearman = sum(spearman)/len(spearman)
    return spearman, train_spearman


def main(config=None, ctx: TrainContext=TrainContext()):
    ctx.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    total_iters = 5
    with wandb.init(project="Antibody Succotash", config=config) as run:
        device = ctx.device
        loss = torch.nn.GaussianNLLLoss()
        iter_scores = []
        mean_iter_scores = []
        run.log({"model": "CNN"})
        X = generate_features()

        for iter in range(total_iters):
            scores: list[float] = []
            for fold in range(5):
                test_data, train_data, test_loader, train_loader, pca = get_test_fold_dataloaders(
                    fold,
                    X,
                    ctx.batch_size,
                    ndims=run.config['pca_reduction']
                )

                # Initialize model, send it to device
                model = ConvModelFullVector(**run.config['model_params'])
                model.to(device)

                # Initialize optimizer (we use Adam here)
                optimizer = torch.optim.Adam(model.parameters(), lr=run.config['lr'])

                epochs: int = run.config['epochs']
                try:
                    spearman: float = float('nan')
                    for t in range(epochs):
                        ctx.epoch = t
                        train_loop(train_loader, model, loss, optimizer, ctx)
                        spearman, train_spearman = test_loop(test_loader, model, loss, train_data, ctx)
                        run.log({"iter": {f"{iter}": {
                            f"fold-{fold}": {
                                "epoch": t,
                                "test_spearman": spearman,
                                "train_spearman": train_spearman
                        }}}})
                    if np.isnan(spearman):  # Set score = -1 if model fails to differentiate between points
                        spearman = -1
                    scores.append(spearman)
                    mean_iter_scores.append(spearman)
                except Exception as e:
                    run.log({"exception": traceback.format_exc()})
                    spearman = -1
                    scores.append(spearman)
                    mean_iter_scores.append(spearman)
                    raise Exception(f"Run {run.id} failed on fold {fold}, iteration {iter}") from e
                finally:
                    run.log({"mean_spearman": np.mean(mean_iter_scores)})

            score = (np.mean(scores) + np.median(scores) + np.min(scores))/3
            mean_score = np.mean(scores)
            run.log({"iter": {f"{iter}":{"spearman": score, "spearman_mean": mean_score}}})
            iter_scores.append(score)
            run.log({"spearman": np.mean(iter_scores)})


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    train_ctx = TrainContext(batch_size=32, device=device)
