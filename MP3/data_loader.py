"""
To create the train, val and test DataLoaders classes
"""
import os

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

from utils import print_infos, show_image
from params import *


NUM_WORKERS = 0  # os.cpu_count()


class MyDataset(Dataset):
    """To load the pickled data file. Class from tutorial 6 of ECSE-551

    img_file: the pickle file containing the images
    label_file: the .csv file containing the labels
    transform: We use it for normalizing images (see above)
    idx: This is a binary vector that is useful for creating training and validation set.

    It return only samples where idx is True.

    Attributes
    ----------
    data
    targets
    transform

    """

    def __init__(
        self,
        img_file_name,
        label_file_name=None,
        transform=None,
        idx=None,
        folder_path=None,
    ):
        img_file_name = f"{folder_path}/{img_file_name}"

        with open(img_file_name, "rb") as img_file:
            self.data = pickle.load(img_file, encoding="bytes")

        if N_CHANNEL == 3:
            # Repeat grey image for 3 channels. To allow the use with RGB models
            self.data = np.repeat(self.data, 3, 1)

        self.data = np.moveaxis(self.data, 1, -1)  # Move channel to last position

        if label_file_name is not None:
            label_file = f"{folder_path}/{label_file_name}"

            self.targets = np.genfromtxt(label_file, delimiter=",", skip_header=1)[
                :, 1:
            ]

        else:
            self.targets = None

        if idx is not None:
            self.targets = self.targets[idx]
            self.data = self.data[idx]

        self.transform = transform

    def __len__(self):
        if self.targets is not None:
            l = len(self.targets)
        else:
            l = self.data.shape[0]
        return l

    def __getitem__(self, index):
        img = self.data[index]
        if self.targets is None:
            target = None
        else:
            target = int(self.targets[index])

        if self.transform is not None:
            # img = Image.fromarray(img.astype('float'), mode='L')
            img_trans = self.transform(img)

        # If not transform specified, transform to torch tensor
        else:
            img_trans = transforms.ToTensor()(img)

        return img_trans, target


def create_dataloaders(
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int = NUM_WORKERS,
    print_ds_infos: bool = False,
):
    print("Creating datasets...")
    # Load the dataset
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize(
                [0.5], [0.5]
            ),  # transforms.Normalize((0.1307,), (0.3081,) #TODO: Find mean and var of dataset
        ]
    )

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    automatic_transforms = weights.transforms()
    # img_transform = automatic_transforms

    data_folder = f"MP3/data"

    train_file = "Train.pkl"
    train_label_file = "Train_labels.csv"
    test_file = "Test.pkl"
    val_size = 0.15

    # Datasets
    no_transform_ds = MyDataset(
        train_file,
        label_file_name=train_label_file,
        transform=None,
        folder_path=data_folder,
    )
    full_train_ds = MyDataset(
        train_file,
        label_file_name=train_label_file,
        transform=img_transform,
        folder_path=data_folder,
    )
    test_ds = MyDataset(test_file, transform=img_transform, folder_path=data_folder)

    # Train/Val Datasets
    train_indices, val_indices, _, _ = train_test_split(
        range(len(full_train_ds)),
        full_train_ds.targets,
        stratify=full_train_ds.targets,
        test_size=val_size,
        random_state=SEED,
    )

    train_ds = MyDataset(
        train_file,
        label_file_name=train_label_file,
        transform=img_transform,
        folder_path=data_folder,
        idx=train_indices,
    )
    val_ds = MyDataset(
        train_file,
        label_file_name=train_label_file,
        transform=img_transform,
        folder_path=data_folder,
        idx=val_indices,
    )

    # Dataloaders
    print("Creating dataloaders...")
    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    # Print information
    if print_ds_infos:
        print(f"--- Full Train dataset ---")
        print_infos(full_train_ds)

        print(f"--- Train dataset ---")
        print_infos(train_ds)

        print(f"--- Val dataset ---")
        print_infos(val_ds)

        print(f"\n--- Test dataset ---")
        print_infos(test_ds)

    return train_dl, val_dl, test_dl
