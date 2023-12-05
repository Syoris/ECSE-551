"""
To create the model
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Literal
import torchvision

from utils import set_seed
from params import *


class Net(nn.Module):
    # This part defines the layers
    def __init__(self):
        super(Net, self).__init__()
        # At first there is only 1 channel (greyscale). The next channel size will be 10.
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Then, going from channel size (or feature size) 10 to 20.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Now let us create some feed foreward layers in the end. Remember the sizes (from 320 to 50)
        self.fc1 = nn.Linear(320, 50)
        # The last layer should have an output with the same dimension as the number of classes
        self.fc2 = nn.Linear(50, 10)

    # And this part defines the way they are connected to each other
    # (In reality, it is our forward pass)
    def forward(self, x):
        # F.relu is ReLU activation. F.max_pool2d is a max pooling layer with n=2
        # Max pooling simply selects the maximum value of each square of size n. Effectively dividing the image size by n
        # At first, x is out input, so it is 1x28x28
        # After the first convolution, it is 10x24x24 (24=28-5+1, 10 comes from feature size)
        # After max pooling, it is 10x12x12
        # ReLU doesn't change the size
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Again, after convolution layer, size is 20x8x8 (8=12-5+1, 20 comes from feature size)
        # After max pooling it becomes 20x4x4
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # This layer is an imaginary one. It simply states that we should see each member of x
        # as a vector of 320 elements, instead of a tensor of 20x4x4 (Notice that 20*4*4=320)
        x = x.view(-1, 320)

        # Feedforeward layers. Remember that fc1 is a layer that goes from 320 to 50 neurons
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)

        # We should put an appropriate activation for the output layer.
        return F.log_softmax(x)


def get_model():
    # Option 1 - Pre trained model
    weights = (
        torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )  # NEW in torchvision 0.13, "DEFAULT" means "best weights available"
    model = torchvision.models.efficientnet_b0(weights=weights).to(DEVICE)

    # Freeze all base layers by setting requires_grad attribute to False
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    # Update the classifier head to suit our problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=10, bias=True).to(DEVICE),
    )

    # # Option 2
    # model = Net()

    return model


def get_optimizer(
    network: Net,
    type: Literal["SGD", "Adam"] = "SGD",
    lr: float = 0.01,
    momentum: float = 0.5,
):
    if type == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
    elif type == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=lr)

    return optimizer


def get_loss_fn(type: Literal["nll", "ce"] = "ce"):
    return nn.CrossEntropyLoss()
