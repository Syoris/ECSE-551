# imports
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# My functions
from data_loader import create_dataloaders
from model import get_model, get_optimizer, get_loss_fn
from training import train_model
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
    img_size = 32

    # Model
    model_name = 'VGG13'

    act_fn = 'ReLu'
    dropout_prob = 0.15

    # Optim
    n_epochs = 50
    optimizer = 'Adam'
    lr = 0.001
    momentum = 0.5

    # Loss
    loss_fn = 'cross_entropy'  # 'cross_entropy', 'nll'


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
        'model_name': model_name,
        'act_fn': act_fn,
        'dropout_prob': dropout_prob,
        # Optim
        'optimizer': optimizer,
        "n_epoch": n_epochs,
        "lr": lr,
        "momentum": momentum,
        # Loss
        "loss_fn": loss_fn
    }
    run["parameters"] = hyperparameters

    # ---- Load Data ---
    train_dl, val_dl, test_dl = create_dataloaders(
        img_size, train_batch_size, test_batch_size, print_ds_infos=False, neptune_run=run
    )

    # --- Train Model ---
    model = get_model(model_type=model_name, neptune_run=run, act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size)

    optimizer = get_optimizer(model, type="Adam")

    loss_fn = get_loss_fn(loss_fn)

    results = train_model(model, train_dl, val_dl, optimizer, loss_fn, n_epochs, run)

    run.stop()

    utils.plot_training_loss(results)
    utils.plot_training_acc(results)

    ...


if __name__ == "__main__":
    train_models()
    ...
