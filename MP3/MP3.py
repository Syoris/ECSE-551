import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# My functions
from data_loader import create_dataloaders
from model import get_model, get_optimizer, get_loss_fn, log_model_info
from training import train_model, predict
import utils
from params import *
import neptune


torch.backends.cudnn.enabled = False


def train_models():
    print(f"------- Training models -------")
    # --- Hyperparameters ---
    # Datasets
    train_batch_size = 64
    test_batch_size = 64
    img_size = 64

    # Model
    model_name = "VGG16"

    act_fn = "ReLu"
    dropout_prob = 0.15

    # Optim
    n_epochs = 10
    optimizer_type = "Adam"
    lr = 0.001
    momentum = 0.5

    # Loss
    loss_fn = "cross_entropy"  # 'cross_entropy', 'nll'

    # --- Setup Run ---
    run_name = utils.get_run_name(model_name)
    print(f"Model: {model_name}\t Neptune run: {run_name}")
    run = neptune.init_run(
        project="MyResearch/ECSE551-MP3",
        api_token=NEPTUNE_API,
        custom_run_id=run_name,
        source_files=["MP3/*.py"],
    )

    # Log hyperparameters
    hyperparameters = {
        "seed": SEED,
        # Dataset
        "img_size": img_size,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        # Model
        "model_name": model_name,
        "act_fn": act_fn,
        "dropout_prob": dropout_prob,
        # Optim
        "optimizer": optimizer_type,
        "n_epoch": n_epochs,
        "lr": lr,
        "momentum": momentum,
        # Loss
        "loss_fn": loss_fn,
    }
    run["parameters"] = hyperparameters

    # ---- Load Data ---
    train_dl, val_dl, _, full_train_dl = create_dataloaders(
        img_size, train_batch_size, test_batch_size, print_ds_infos=False, neptune_run=run
    )

    # --- Train Model ---
    model = get_model(
        model_type=model_name, act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size
    )
    log_model_info(model, img_size, run)

    optimizer = get_optimizer(model, type=optimizer_type, lr=lr, momentum=momentum)

    loss_fn = get_loss_fn(loss_fn)

    results = train_model(model, train_dl, val_dl, optimizer, loss_fn, n_epochs, run)

    run.stop()

    utils.plot_training_loss(results)
    utils.plot_training_acc(results)

    ...


def load_run(
    run_id: str, retrain: bool
) -> Tuple[nn.Module, List[torch.utils.data.DataLoader]]:  # (model, [full_train_dl , test_dl])
    """Load all the info from a previous run. If the model is not to be retrained, load the models'
    weights and optimizer state (TODO)

    Args:
        run_id (str): _description_
        retrain (bool): If the model is to be retrained from 0 on the full dataset (train+val)

    Returns:
        Tuple[nn.Module, List[torch.utils.data.DataLoader]]: _description_
    """
    # --- Load data ---
    print(f"Loading run from Neptune: {run_id}")

    run = neptune.init_run(
        project="MyResearch/ECSE551-MP3", with_id=run_id, api_token=NEPTUNE_API, mode="read-only"
    )

    model_id = run["sys/custom_run_id"].fetch()
    model_params = run["parameters"].fetch()
    model_type = model_id.split('_')[0]

    # --- Hyperparameters ---
    # Datasets
    train_batch_size = model_params["train_batch_size"]
    test_batch_size = model_params["test_batch_size"]
    img_size = model_params["img_size"]

    # Model
    act_fn = model_params["act_fn"]
    dropout_prob = model_params["dropout_prob"]

    # Optim
    n_epochs = model_params["n_epoch"]
    optimizer_type = model_params["optimizer"]
    lr = model_params["lr"]
    momentum = model_params["momentum"]

    # Loss
    loss_fn = model_params["loss_fn"]

    # --- Model ---
    model = get_model(
        model_type=model_type, act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size
    )
    if not retrain:
        weights_path = Path("MP3/models") / model_id / "model.pth"
        model.load_state_dict(torch.load(weights_path))

    # --- Optim ---
    optimizer = get_optimizer(model, type=optimizer_type, lr=lr, momentum=momentum)

    # --- Loss ---
    loss_fn = get_loss_fn(loss_fn)

    # --- Datasets ---
    train_dl, val_dl, test_dl, full_train_dl = create_dataloaders(
        img_size, train_batch_size, test_batch_size, print_ds_infos=False, neptune_run=None
    )

    run.stop()
    print('Done loading')

    return (
        model_id,
        model,
        [train_dl, full_train_dl, val_dl, test_dl],
        optimizer,
        loss_fn,
        n_epochs,
    )


def continue_training(run_id, n_add_epochs):
    (
        model_id,
        model,
        (train_dl, full_train_dl, val_dl, test_dl),
        optimizer,
        loss_fn,
        n_epochs,
    ) = load_run(run_id, retrain=False)

    print(f'Continuing training of: {run_id}, {model_id}')
    print(f'\tTraining from {n_epochs} for {n_add_epochs} more epochs')

    # run = neptune.init_run(
    #     project="MyResearch/ECSE551-MP3", with_id=run_id, api_token=NEPTUNE_API, mode="read-only"
    # )

    # run_name = utils.get_run_name(model_id.split('_')[0])
    # print(f'New run: {run_name}')
    train_run = neptune.init_run(
        project="MyResearch/ECSE551-MP3",
        api_token=NEPTUNE_API,
        with_id=run_id,
        source_files=["MP3/*.py"],
    )
    train_run["parameters/n_epoch"] = n_add_epochs + n_epochs

    train_model(
        model,
        train_dl,
        val_dl,
        optimizer,
        loss_fn,
        n_add_epochs,
        train_run,
        start_epoch=n_epochs + 1,
    )
    train_run.stop()


def test_model(run_id, retrain=True):
    (
        model_id,
        model,
        (train_dl, full_train_dl, val_dl, test_dl),
        optimizer,
        loss_fn,
        n_epochs,
    ) = load_run(run_id, retrain=retrain)

    if retrain:
        model_id += '_Full'
        print(f'Starting new training: {model_id}')
        train_run = neptune.init_run(
            project="MyResearch/ECSE551-MP3",
            api_token=NEPTUNE_API,
            custom_run_id=model_id,
            source_files=["MP3/*.py"],
        )
        train_model(model, full_train_dl, val_dl, optimizer, loss_fn, n_epochs, train_run)
        train_run.stop()

    y_test = predict(model, test_dl)

    pred_df = pd.DataFrame(y_test, columns=['class'])
    pred_df.index.name = 'id'
    pred_save_path = Path('MP3/predictions') / f'{model_id}.csv'
    pred_df.to_csv(pred_save_path)
    print(f'Predictions saved to {pred_save_path}')

    ...


def add_test_acc(run_id, test_acc: float):
    """To add test accuract from Kaggle to Neptune.

    test_acc: In decimal [0, 1]
    """
    print(f"Adding test acc to: {run_id}")

    run = neptune.init_run(project="MyResearch/ECSE551-MP3", with_id=run_id, api_token=NEPTUNE_API)

    run['test_acc'] = test_acc * 100
    run.stop()


if __name__ == "__main__":
    # train_models()

    run_id = "MP3-74"
    n_add_epochs = 2
    continue_training(run_id, n_add_epochs)

    # run_id = "MP3-66"
    # n_epochs = 10
    # test_model(run_id, n_epochs)

    # # Add test acc to neptune
    # run_id = "MP3-69"
    # test_acc = 0.89666
    # add_test_acc(run_id, test_acc)
    # ...
