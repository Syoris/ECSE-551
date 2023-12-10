import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import chain
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from params import *
import neptune


# ------- General utils -------
def set_seed(seed: int = 0):
    """Sets seed for PyTorch"""
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


# ------- Datasets utils -------
def print_infos(dataset, img_idx=None):
    """To print information about the dataset.

    Shows image at img_idx or a random one if None.

    Args:
        img_idx (int, optional): Image to show. Defaults to None.
    """
    img_shape = dataset[0][0].shape

    # General information
    print(f"Information about the dataset:")
    print(f"\tNumber of samples: {dataset.data.shape[0]}")
    print(f"\tFeature space: {img_shape}")

    if dataset.targets is not None:
        classes, classes_count = np.unique(dataset.targets, return_counts=True)
        print(
            f"\tDifferent classes: {classes}. Their proportion: {classes_count/classes_count.sum()}"
        )

    if img_idx is not None:
        print(f"Image {img_idx}")
        plt.imshow(dataset.data[img_idx, 0], cmap="gray", vmin=0, vmax=1)

    print()


def compute_mean_std(dataset: np.ndarray):
    print(f"\nComputing mean and variance...")
    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])

    image_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    for inputs in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 1, 2])
        psum_sq += (inputs**2).sum(axis=[0, 1, 2])

    count = len(dataset) * 28 * 28

    # mean and STD
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    print("- mean: {:.4f}".format(total_mean.item()))
    print("- std:  {:.4f}".format(total_std.item()))


def show_image(img, vmin=0, vmax=1, ax=None):
    """To show the image

    Args:
        img (Image): Image to show
        vmin (int, optional): Min value of the scale. Defaults to 0.
        vmax (int, optional): Max value of the scale. Defaults to 1.
    """
    # plt.imshow(img.cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)

    if ax is not None:
        ax.imshow(img[0], cmap="gray", vmin=vmin, vmax=vmax)

    else:
        plt.imshow(img[0], cmap="gray", vmin=vmin, vmax=vmax)


# ------- Plotting -------
def plot_training_loss(results: dict):
    train_counter = list(chain(*results["train_log_counter"]))
    train_losses = list(chain(*results["train_loss"]))

    val_counter = results["val_log_counter"]
    val_losses = results["val_loss"]

    # Plot training progression
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(val_counter, val_losses, color="red")
    plt.legend(["Train Loss", "Validation Loss"], loc="upper right")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("Loss")
    plt.title("Loss - Training Progression")
    plt.show(block=False)


def plot_training_acc(results: dict):
    train_counter = list(chain(*results["train_log_counter"]))
    train_acc = list(chain(*results["train_acc"]))

    val_counter = results["val_log_counter"]
    val_acc = results["val_acc"]

    # Plot training progression
    fig = plt.figure()
    plt.plot(train_counter, train_acc, color="blue")
    plt.scatter(val_counter, val_acc, color="red")
    plt.legend(["Train Acc", "Validation Acc"], loc="upper right")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("Acc")
    plt.title("Accuracy - Training Progression")
    plt.show(block=False)


# ------- Neptune run utils -------
def get_run_data(run_id: str) -> dict:
    """To load the data from a run

    Args:
        run_id (str): Run id in Neptune

    Returns:
        dict: Run's data. {key: pd.DataFrame}
            Dataframe cols: index, step, value
            Keys: train_loss, train_acc, val_loss, val_acc
    """
    # --- Load data ---
    print(f"Loading run from Neptune: {run_id}")

    run = neptune.init_run(
        project="MyResearch/ECSE551-MP3", with_id=run_id, api_token=NEPTUNE_API, mode="read-only"
    )

    # model_id = run["sys/custom_run_id"].fetch()
    # model_params = run["parameters"].fetch()
    # model_type = model_id.split('_')[0]

    # --- Parse data ---
    run_data_dict = {}
    training_loss = run['training/loss'].fetch_values()
    training_acc = run['training/acc'].fetch_values()

    val_loss = run['val/loss'].fetch_values()
    val_acc = run['val/acc'].fetch_values()

    run_data_dict['train_loss'] = training_loss[['step', 'value']]
    run_data_dict['train_acc'] = training_acc[['step', 'value']]
    run_data_dict['val_loss'] = val_loss[['step', 'value']]
    run_data_dict['val_acc'] = val_acc[['step', 'value']]

    return run_data_dict


def generate_run_name(model_name: str):
    from datetime import datetime

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")  # returns current date in YYYY-MM-DD format

    return f"{model_name}_{timestamp}"
