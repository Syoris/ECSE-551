"""
To create the model
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Literal
import torchvision
from torchinfo import summary

from utils import set_seed
from params import *
import neptune
from neptune.types import File


def compute_img_size(in_size, filter_size, stride, padding, pooling: bool):
    if padding == "same":
        img_size = in_size

    else:
        img_size = int((in_size - filter_size) / stride + 1)

    if pooling:
        img_size /= 2

    return int(img_size)


class Net2(nn.Module):
    # This part defines the layers
    def __init__(self, input_size: int = 1, output_size: int = 10, dropout_prob:float =0.0):
        """Our custom CNN

        Args:
            input_size (int): Number of channel of for input
            output_size (int): Number of classes
        """
        super(Net, self).__init__()

        # Parameters
        act_function = nn.ReLU()
        final_act_function = nn.LogSoftmax(dim=1)
        dropout_rate = nn.Dropout(p=dropout_prob)
        pool = nn.MaxPool2d(kernel_size=(2, 2))

        # ------- Conv -------
        # Conv1: 1 channel(greyscale) to 10 channnels. Kernel=5, stride=1
        in_channels = input_size
        out_channels = 10
        kernel_size = 5
        padding = "valid"  # same: With 0 padding, image is same size, valid: no padding
        stride = 1
        in_img_size = IMG_SIZE
        pooling = True
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv1_act = act_function
        self.conv1_pool = pool

        out_img_size = compute_img_size(in_img_size, kernel_size, stride, padding, pooling)

        # Conv2: 10 channels to 20 channnels. Kernel=5, stride=1
        in_channels = out_channels
        out_channels = 20
        kernel_size = 5
        padding = "valid"  # same: With 0 padding, image is same size, valid: no padding
        stride = 1
        in_img_size = out_img_size
        pooling = True
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_act = act_function
        self.conv2_pool = pool

        out_img_size = compute_img_size(in_img_size, kernel_size, stride, padding, pooling)

        # # Conv forward pass
        # self.conv_seq_1 = nn.Sequential(self.conv1, act_function, pool) 
        # self.conv_seq_2 = nn.Sequential(self.conv2, act_function, pool) 

        # self.conv_fw = nn.Sequential(self.conv_seq_1, self.conv_seq_2)

        # ------- Seq -------
        self.fc_in_size = out_img_size**2 * out_channels  # Input size to fully connected layers
        self.fc1 = nn.Linear(self.fc_in_size, 50)
        self.fc1_act = act_function
        self.fc1_drop = dropout_rate

        # The last layer size the same as the num of classes
        self.fc2 = nn.Linear(50, output_size)
        self.fc2_act = final_act_function

        # # Seq forward pass
        # self.fc_seq_1 = nn.Sequential(self.fc1, act_function, dropout_rate)
        # self.fc_seq_2 = nn.Sequential(self.fc2, dropout_rate, final_act_function)
        
        # self.fc_fw = nn.Sequential(self.fc_seq_1, self.fc_seq_2)

    # And this part defines the way they are connected to each other
    # (In reality, it is our forward pass)
    def forward(self, x):
        # # Convolutional layer foward pass
        # x = self.conv_seq_1(x)
        # x = self.conv_seq_1(x)

        # # Imaginary layer
        # x = x.view(-1, self.fc_in_size)

        # # Fully connected layers forward pass
        # x = self.fc_fw(x)

        # ----- Conv -----
        # Conv1
        x = self.conv1_act(self.conv1_pool(self.conv1(x)))

        # Conv2
        x = self.conv2_act(self.conv2_pool(self.conv2(x)))

        # ----- Fully Connected -----
        # Imaginary layer
        x = x.view(-1, self.fc_in_size)

        # FC1
        # x = self.fc1_drop(self.fc1_act(self.fc1(x)))
        x = self.fc1_act(self.fc1(x))


        # FC2
        x = self.fc2(x)

        return F.log_softmax(x)

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
    # (In reality, it is our foreward pass)
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
    
def get_efficientnet_b0():
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

    return model


def get_my_net():
    set_seed()
    model = Net()

    return model


def get_model(model_type: Literal["net"], neptune_run: neptune.Run):
    # model = get_efficientnet_b0()
    if model_type == "net":
        model = get_my_net()

    # summary(model)
    model_info = summary(
        model,
        input_size=(
            TRAIN_BATCH_SIZE,
            N_CHANNEL,
            IMG_SIZE,
            IMG_SIZE,
        ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    neptune_run["model/type"] = model_type
    neptune_run["model/summary"].upload(File.from_content(str(model_info)))

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
