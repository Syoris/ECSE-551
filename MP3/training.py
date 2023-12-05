"""
Functions to train the model
"""
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import Dict, List
from tqdm.auto import tqdm


def train_model(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    """To train a model for the number of epochs specified

    Returns:
      A dictionary with the metrics. Each metric has a value in a list for each epoch.
      For training metrics, the values is a list with the value of each batch.
      The first value of validation metric is pre-training so the size of the list is 1+epochs

      Form:
        {train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]}

      For example if training for epochs=2:
              {train_loss: [[2.0616, ..., 1.251], [1.251, ..., 1.059]],
                train_acc: [0.3945, 0.3945],
                val_loss: [1.2641, 1.5706],
                val_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # To keep track of the progress
    train_losses = []
    train_counter = []
    val_losses = []
    val_accs = []
    val_counter = [0]  # When the test function is called

    # Create a writer with all default settings
    writer = SummaryWriter()

    # val_step()
    for epoch in tqdm(range(epochs)):
        # train
        train_step()

        # val
        val_step()

        # Print

        # To TensorBoard

        ...

    writer.close()
    return results


def train_step():
    ...


def val_step():
    ...
