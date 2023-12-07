import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import chain


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


def set_seed(seed: int = 0):
    """Sets seed for PyTorch"""
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


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


def get_run_name(model_name: str):
    from datetime import datetime

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime(
        "%y%m%d_%H%M"
    )  # returns current date in YYYY-MM-DD format

    return f"{model_name}_{timestamp}"
