"""
Functions to train the model
"""
from torch.utils.tensorboard import SummaryWriter


def train_model(model, optimizer):
    # To keep track of the progress
    train_losses = []
    train_counter = []
    val_losses = []
    val_accs = []
    val_counter = [0]  # When the test function is called

    # Create a writer with all default settings
    writer = SummaryWriter()

    ...
