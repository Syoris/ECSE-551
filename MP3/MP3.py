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
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

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

    # --- Setup Run ---
    model_name = "LeNet5"

    run_name = utils.get_run_name(model_name)
    print(f"Model: {model_name}\t Neptune run: {run_name}")
    run = neptune.init_run(
        project="MyResearch/ECSE551-MP3",
        api_token=NEPTUNE_API,
        custom_run_id=run_name,
        source_files=["MP3/*.py"],
    )

    # ---- Load Data ---
    train_dl, val_dl, test_dl = create_dataloaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, print_ds_infos=False, neptune_run=run
    )

    # --- Train Model ---
    hyperparameters = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "n_epoch": N_EPOCHS,
        "lr": LR,
        "momentum": MOMENTUM,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
    }
    run["parameters"] = hyperparameters

    model = get_model(model_type=model_name, neptune_run=run)

    optimizer = get_optimizer(model, type="Adam")

    loss_fn = get_loss_fn()

    results = train_model(model, train_dl, val_dl, optimizer, loss_fn, N_EPOCHS, run)

    run.stop()

    utils.plot_training_loss(results)
    utils.plot_training_acc(results)

    ...


if __name__ == "__main__":
    train_models()
    ...
