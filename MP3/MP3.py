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
from torchinfo import summary

# My functions
from data_loader import create_dataloaders
from model import get_model, get_optimizer, get_loss_fn
from training import train_model
from utils import set_seed, plot_training_acc, plot_training_loss
from params import *

torch.backends.cudnn.enabled = False


def train_models():
    print(f"------- Training models -------")
    train_dl, val_dl, test_dl = create_dataloaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, print_ds_infos=True
    )

    model = get_model()

    # TODO: Move to get_model
    # summary(model)
    summary(
        model,
        input_size=(
            TRAIN_BATCH_SIZE,
            1,
            28,
            28,
        ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    optimizer = get_optimizer(model, type="Adam")

    loss_fn = get_loss_fn()

    results = train_model(model, train_dl, val_dl, optimizer, loss_fn, N_EPOCHS)

    plot_training_loss(results)
    plot_training_acc(results)

    ...


if __name__ == "__main__":
    train_models()
    ...
