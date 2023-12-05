import matplotlib.pyplot as plt
import numpy as np
import torch


def print_infos(dataset, img_idx=None):
    """To print information about the dataset.

    Shows image at img_idx or a random one if None.

    Args:
        img_idx (int, optional): Image to show. Defaults to None.
    """

    # General information
    print(f"Information about the dataset:")
    print(f"\tNumber of samples: {dataset.data.shape[0]}")
    print(f"\tFeature space: {dataset.data.shape[-2]}*{dataset.data.shape[-1]}")

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
