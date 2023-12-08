"""
Functions to train the model
"""
import torch
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from params import *
import numpy as np
import neptune
from pathlib import Path

PRINT_TRAINING = False


def train_model(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    run: neptune.Run,
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
    train_n_samples = len(train_dl.dataset)

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_log_counter": [],
        "val_loss": [],
        "val_acc": [],
        "val_log_counter": [],
    }

    # Neptune run name
    run_name = run["sys/custom_run_id"].fetch()

    # Validation for random weights
    val_loss, val_acc, val_log_counter = val_step(
        model=model,
        dataloader=val_dl,
        loss_fn=loss_fn,
        epoch_num=0,
        train_n_samples=train_n_samples,
        run=run,
    )
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)
    results["val_log_counter"].append(val_log_counter)

    for epoch in range(1, epochs + 1):
        print(f"####### Epoch {epoch} #######")

        # train
        train_loss, train_acc, train_log_counter = train_step(
            model=model,
            dataloader=train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch_num=epoch,
            run=run,
        )

        # val
        val_loss, val_acc, val_log_counter = val_step(
            model=model,
            dataloader=val_dl,
            loss_fn=loss_fn,
            epoch_num=epoch,
            train_n_samples=train_n_samples,
            run=run,
        )
        # Save model
        try:
            folder_path = Path(f"MP3/models/{run_name}")
            folder_path.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), folder_path / "model.pth")
            torch.save(optimizer.state_dict(), folder_path / "optimizer.pth")
        except Exception as err:
            print(f"Error saving model")

        # To results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_log_counter"].append(train_log_counter)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_log_counter"].append(val_log_counter)

        ...

    run["final_acc"] = results["val_acc"][-1] * 100.0

    print(f"-----------------------------")
    print(f'Final val acc: {results["val_acc"][-1] * 100. :.2f}%')
    print(f"-----------------------------")

    return results


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_num: int,
    run: neptune.Run,
) -> Tuple[List[float], List[float]]:
    """Train the model for an epoch.

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Training dataloader
        loss_fn (torch.nn.Module): Loss function to min
        optimizer (torch.optim.Optimizer): Optimizer of loss fn
        epoch_num (int): Epoch num. Used for logging

    Returns:
        Tuple[List[float], List[float], List[int]]: (training_loss, training_acc, train_log_counter)
        Where training_loss and training_acc are list with their value for each log_interval
        train_log_counter: Total # of samples seen when the other metrics were computed
    """

    # Model in train mode
    model.train()

    train_loss = []
    train_acc = []
    train_log_counter = []

    # For each batch
    for batch_idx, (X, y) in enumerate(tqdm(dataloader)):
        # Send data to target device
        X, y = X.to(DEVICE), y.to(DEVICE)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Loss
        loss = loss_fn(y_pred, y)  # negative log liklhood loss

        # 3. Set gradients to 0
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()  # Compute

        # 5. Update weights
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc = (y_pred_class == y).sum().item() / len(y_pred)

        # 6. Prints
        if batch_idx % LOG_INTERVAL == 0:
            n_samples = len(dataloader.dataset)
            n_batch = len(dataloader)
            prev_count = (epoch_num - 1) * n_samples  # Total count from prev epochs
            curr_count = (
                batch_idx * TRAIN_BATCH_SIZE
            )  # Num of samples seen for current epoch
            counter_val = curr_count + prev_count  # Total of sample seen

            train_loss.append(loss.item())
            train_log_counter.append(counter_val)  # Total num of samples seen
            train_acc.append(acc)

            run["training/loss"].append(value=loss.item(), step=counter_val/n_samples)
            run["training/acc"].append(value=acc, step=counter_val/n_samples)

            # e.g. Train Epoch: 2 [8320/51000 (16%)]    Loss: 1.133699  Acc: 42.5%
            if PRINT_TRAINING:
                print(
                    f"Train Epoch: {epoch_num} [{curr_count}/{n_samples} ({100. * batch_idx / n_batch:.0f}%){']':<5} Loss: {loss.item():<10.6f} Acc: {acc:.6f}"
                )

    return (np.array(train_loss), np.array(train_acc), np.array(train_log_counter))


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    epoch_num: int,
    train_n_samples: int,
    run: neptune.Run,
) -> Tuple[float, float, int]:
    """Performs a forward pass on the validation dataset

    Args:
        model (torch.nn.Module): Model to validate
        dataloader (torch.utils.data.DataLoader): Validation dataloader
        loss_fn (torch.nn.Module): Loss function to use
        epoch_num (int): Epoch number

    Returns:
        Tuple[float, float, int]: (val_loss, val_acc, val_log_counter)
            val_log_counter: # of samples seen when validation is performed
    """
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for X, y in dataloader:
            # Send data to device
            X, y = X.to(DEVICE), y.to(DEVICE)

            # 1. Forward pass
            val_pred = model(X)

            # 2. Loss
            loss = loss_fn(val_pred, y)
            val_loss += loss.item()

            # 3. Accuracy
            pred_labels = val_pred.argmax(dim=1)
            val_acc += (pred_labels == y).sum().item() / len(pred_labels)

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    val_log_counter = epoch_num * train_n_samples

    # Add to run
    run["val/loss"].append(value=val_loss, step=epoch_num)
    run["val/acc"].append(value=val_acc, step=epoch_num)

    print(
        f"Validation set: Avg. Loss: {val_loss:<10.4f} Avg. Acc: {val_acc*100:.2f}%\n"
    )

    return (np.array(val_loss), np.array(val_acc), np.array(val_log_counter))
