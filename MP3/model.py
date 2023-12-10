"""
To create the model
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import flatten

from typing import Literal, List
import torchvision
from torchinfo import summary

from utils import set_seed
from params import *
import neptune
from neptune.types import File


ACT_FN_DICT = {
    'ReLu': nn.ReLU(),
    'tanh': nn.Tanh(),
}


def compute_img_size(in_size, filter_size, stride, padding, pooling: bool):
    if isinstance(padding, str):
        if padding == "same":
            img_size = in_size

        else:
            img_size = int((in_size - filter_size) / stride + 1)

    else:
        img_size = (in_size - filter_size + 2 * padding) / stride + 1

    if pooling:
        img_size /= 2

    return int(img_size)


def make_conv_layers(layers_list, act_fn, use_batch_norm: bool = False) -> List[nn.Module]:
    layers: List[nn.Module] = []
    in_channels = 1

    for layer in layers_list:
        if layer['type'] == 'avg':
            act_fn = layer['act']

            kernel_size = layer['f']
            stride = layer['s']
            padding = layer['p']

            avgPool = nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride)
            layers += [avgPool, act_fn]

        elif layer['type'] == 'max_pool':
            kernel_size = layer['f']

            avgPool = nn.MaxPool2d(kernel_size=kernel_size)
            layers += [avgPool]

        elif layer['type'] == 'conv':
            out_channels = int(layer['out_ch'])
            kernel_size = layer['f']
            stride = layer['s']
            padding = layer['p']
            conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
            )
            if use_batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), act_fn]
            else:
                layers += [conv2d, act_fn]

            in_channels = out_channels

        else:
            raise ValueError(f"Invalid layer type: {layer['type']}")

    return layers


class MyNet(nn.Module):
    # This part defines the layers
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 10,
        dropout_prob: float = 0.15,
        img_size: int = 32,
    ):
        """Our custom CNN

        Args:
            input_size (int): Number of channel of for input
            output_size (int): Number of classes
        """
        super(MyNet, self).__init__()

        # Parameters
        act_function = nn.ReLU()
        final_act_function = nn.LogSoftmax()
        dropout_rate = nn.Dropout(p=dropout_prob)
        pool = nn.MaxPool2d(kernel_size=(2, 2))

        # ------- Conv -------
        # Conv1: 1 channel(greyscale) to 8 channnels. Kernel=5, stride=1
        in_channels = input_size
        out_channels = 8
        kernel_size = 5
        padding = "valid"  # same: With 0 padding, image is same size, valid: no padding
        stride = 1
        in_img_size = 28
        pooling = False
        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        conv1_act = act_function

        self.conv1_seq = nn.Sequential(conv1, nn.ReLU())

        out_img_size = compute_img_size(in_img_size, kernel_size, stride, padding, pooling)

        # Conv2: 10 channels to 20 channnels. Kernel=5, stride=1
        in_channels = out_channels
        out_channels = 16
        kernel_size = 5
        padding = "valid"  # same: With 0 padding, image is same size, valid: no padding
        stride = 1
        in_img_size = out_img_size
        pooling = True
        conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        conv2_act = act_function
        conv2_pool = pool

        self.conv2_seq = nn.Sequential(conv2, nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))

        out_img_size = compute_img_size(in_img_size, kernel_size, stride, padding, pooling)

        # Conv3
        in_channels = out_channels
        out_channels = 32
        kernel_size = 5
        padding = "valid"  # same: With 0 padding, image is same size, valid: no padding
        stride = 1
        in_img_size = out_img_size
        pooling = True
        conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        conv3_act = act_function
        conv3_pool = pool

        self.conv3_seq = nn.Sequential(conv3, nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))

        out_img_size = compute_img_size(in_img_size, kernel_size, stride, padding, pooling)

        # # Conv forward pass
        # self.conv_seq_2 = nn.Sequential(self.conv2, act_function, pool)

        self.conv_fw = nn.Sequential(self.conv1_seq, self.conv2_seq, self.conv3_seq)

        # ------- Seq -------
        self.fc_in_size = out_img_size**2 * out_channels  # Input size to fully connected layers
        in_size = self.fc_in_size
        out_size = 256
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc1_act = act_function
        self.fc1_drop = dropout_rate
        self.fc1_seq = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(p=dropout_prob))

        in_size = out_size
        out_size = 64
        fc2 = nn.Linear(in_size, out_size)
        fc2_act = act_function
        fc2_drop = dropout_rate
        self.fc2_seq = nn.Sequential(fc2, nn.ReLU(), nn.Dropout(p=dropout_prob))

        # The last layer size the same as the num of classes
        in_size = out_size
        out_size = output_size
        fc3 = nn.Linear(in_size, out_size)
        fc3_act = final_act_function
        self.fc3_seq = nn.Sequential(fc3, nn.LogSoftmax())

        # # Seq forward pass
        self.fc_fw = nn.Sequential(self.fc1_seq, self.fc2_seq, self.fc3_seq)

    # And this part defines the way they are connected to each other
    # (In reality, it is our forward pass)
    def forward(self, x):
        # ----- Conv -----
        out = self.conv_fw(x)

        # ----- Fully Connected -----
        # Imaginary layer
        out = out.view(-1, self.fc_in_size)

        out = self.fc_fw(out)

        return x


class VGG11(nn.Module):
    # This part defines the layers
    def __init__(
        self,
        input_size: int = 1,
        n_classes: int = 10,
        dropout_prob: float = 0.15,
        img_size: int = 32,
    ):
        """Our implementation of VGG16"""
        super(VGG11, self).__init__()

        # Parameters
        act_function = nn.ReLU()
        final_act_function = nn.LogSoftmax()
        dropout_rate = nn.Dropout(p=dropout_prob)
        pool = nn.MaxPool2d(kernel_size=(2, 2))

        # ------- Conv -------
        conv_layers_list = [64, "M", 128, "M", 256, 256]
        layers: List[nn.Module] = []
        in_channels = 1

        for layer in conv_layers_list:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = int(layer)
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_channels

        self.conv_fw = nn.Sequential(*layers)

        # --- Avg Pool ---
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ------- FC -------
        fc1 = nn.Linear(256 * 7 * 7, 4096)
        fc2 = nn.Linear(4096, 2048)
        fc3 = nn.Linear(2048, n_classes)

        self.fc_fw = nn.Sequential(
            fc1,
            nn.ReLU(True),
            nn.Dropout(p=dropout_prob),
            fc2,
            nn.ReLU(True),
            nn.Dropout(p=dropout_prob),
            fc3,
            nn.LogSoftmax(),
        )

    # And this part defines the way they are connected to each other
    # (In reality, it is our forward pass)
    def forward(self, x):
        # ----- Conv -----
        out = self.conv_fw(x)
        out = self.avgpool(out)

        # ----- Fully Connected -----
        # Imaginary layer
        # out = out.view(-1, self.fc_in_size)
        out = flatten(out, 1)

        out = self.fc_fw(out)

        return out


class VGG13(nn.Module):
    # This part defines the layers
    def __init__(
        self,
        n_classes: int = 10,
        dropout_prob: float = 0.15,
        act_fn: nn.Module = nn.ReLU(),
        img_size: int = 32,
    ):
        """Our implementation of VGG16"""
        super(VGG13, self).__init__()

        if img_size == 32:
            end_size = 1
        elif img_size == 64:
            end_size = 2
        else:
            raise RuntimeError(f"Invalid image size: {img_size} for VGG16 class")

        # ------- Conv -------
        # # input: 32x32
        # VGG13:  [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        conv_layers_list = [
            {'type': 'conv', 'out_ch': 64, 'f': 3, 's': 1, 'p': 1},  # to 32 x 32 x 64
            {'type': 'conv', 'out_ch': 64, 'f': 3, 's': 1, 'p': 1},  # to 32 x 32 x 64
            {'type': 'max_pool', 'f': 2},  # to 16 x 16 x 64
            {'type': 'conv', 'out_ch': 128, 'f': 3, 's': 1, 'p': 1},  # to 16 x 16 x 128
            {'type': 'conv', 'out_ch': 128, 'f': 3, 's': 1, 'p': 1},  # to 16 x 16 x 128
            {'type': 'max_pool', 'f': 2},  # to 8  x 8  x 128
            {'type': 'conv', 'out_ch': 256, 'f': 3, 's': 1, 'p': 1},  # to 8  x 8  x 256
            {'type': 'conv', 'out_ch': 256, 'f': 3, 's': 1, 'p': 1},  # to 8  x 8  x 256
            {'type': 'max_pool', 'f': 2},  # to 4  x 4  x 256
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 4  x 4  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 4  x 4  x 512
            {'type': 'max_pool', 'f': 2},  # to 2  x 2  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 2  x 2  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 2  x 2  x 512
            {'type': 'max_pool', 'f': 2},  # to 1  x 1  x 512
        ]

        layers = make_conv_layers(conv_layers_list, act_fn, use_batch_norm=True)
        self.conv_fw = nn.Sequential(*layers)

        # --- Avg Pool ---
        ...

        # ------- FC -------
        fc1 = nn.Linear(end_size * end_size * 512, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, n_classes)

        self.fc_fw = nn.Sequential(
            fc1,
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            fc2,
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            fc3,
            nn.LogSoftmax(),
        )

    def forward(self, x):
        # ----- Conv -----
        out = self.conv_fw(x)

        # ----- Fully Connected -----
        out = flatten(out, 1)

        out = self.fc_fw(out)

        return out


class VGG16(nn.Module):
    # This part defines the layers
    def __init__(
        self,
        n_classes: int = 10,
        dropout_prob: float = 0.15,
        act_fn: nn.Module = nn.ReLU(),
        img_size: int = 32,
    ):
        """Our implementation of VGG16"""
        super(VGG16, self).__init__()

        if img_size == 32:
            end_size = 1
        elif img_size == 64:
            end_size = 2
        else:
            raise RuntimeError(f"Invalid image size: {img_size} for VGG16 class")

        # ------- Conv -------
        # # input: 64x64
        # VGG16:  [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        conv_layers_list = [
            {'type': 'conv', 'out_ch': 64, 'f': 3, 's': 1, 'p': 1},  # to 64 x 64 x 64
            {'type': 'conv', 'out_ch': 64, 'f': 3, 's': 1, 'p': 1},  # to 64 x 64 x 64
            {'type': 'max_pool', 'f': 2},  # to 16 x 16 x 64
            {'type': 'conv', 'out_ch': 128, 'f': 3, 's': 1, 'p': 1},  # to 32 x 32 x 128
            {'type': 'conv', 'out_ch': 128, 'f': 3, 's': 1, 'p': 1},  # to 32 x 32 x 128
            {'type': 'max_pool', 'f': 2},  # to 8  x 8  x 128
            {'type': 'conv', 'out_ch': 256, 'f': 3, 's': 1, 'p': 1},  # to 16  x 16  x 256
            {'type': 'conv', 'out_ch': 256, 'f': 3, 's': 1, 'p': 1},  # to 16  x 16  x 256
            {'type': 'conv', 'out_ch': 256, 'f': 3, 's': 1, 'p': 1},  # to 16  x 16  x 256
            {'type': 'max_pool', 'f': 2},  # to 4  x 4  x 256
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 8  x 8  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 8  x 8  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 8  x 8  x 512
            {'type': 'max_pool', 'f': 2},  # to 2  x 2  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 4  x 4  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 4  x 4  x 512
            {'type': 'conv', 'out_ch': 512, 'f': 3, 's': 1, 'p': 1},  # to 4  x 4  x 512
            {'type': 'max_pool', 'f': 2},  # to 2  x 2  x 512
        ]

        layers = make_conv_layers(conv_layers_list, act_fn, use_batch_norm=True)
        self.conv_fw = nn.Sequential(*layers)

        # ------- FC -------
        fc1 = nn.Linear(end_size * end_size * 512, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, n_classes)

        self.fc_fw = nn.Sequential(
            fc1,
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            fc2,
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            fc3,
            nn.LogSoftmax(),
        )

    # And this part defines the way they are connected to each other
    # (In reality, it is our forward pass)
    def forward(self, x):
        # ----- Conv -----
        out = self.conv_fw(x)

        # ----- Fully Connected -----
        # Imaginary layer
        # out = out.view(-1, self.fc_in_size)
        out = flatten(out, 1)

        out = self.fc_fw(out)

        return out


class LeNet5(nn.Module):
    # This part defines the layers
    def __init__(
        self,
        n_classes: int = 10,
        dropout_prob: float = 0.15,
        act_fn: nn.Module = nn.ReLU(),
        img_size: int = 32,
    ):
        """Our implementation of LeNet5"""
        if img_size != 32:
            raise ValueError(f'Invalid image size for LeNet5. Only 32x32 images supported.')

        super(LeNet5, self).__init__()

        # ------- Conv -------
        # input: 32x32x1
        conv_layers_list = [
            {'type': 'conv', 'out_ch': 6, 'f': 5, 's': 1, 'p': 'valid'},  # to 28 x 28 x 6
            {'type': 'max_pool', 'f': 2},  # to 14 x 14 x 6
            {'type': 'conv', 'out_ch': 16, 'f': 5, 's': 1, 'p': 'valid'},  # to 10 x 10 x 16
            {'type': 'max_pool', 'f': 2},  # to 05 x 05 x 16
        ]

        layers = make_conv_layers(conv_layers_list, act_fn)
        self.conv_fw = nn.Sequential(*layers)

        # ------- FC -------
        fc1 = nn.Linear(400, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, n_classes)

        self.fc_fw = nn.Sequential(
            fc1,
            nn.ReLU(True),
            nn.Dropout(p=dropout_prob),
            fc2,
            nn.ReLU(True),
            nn.Dropout(p=dropout_prob),
            fc3,
            nn.LogSoftmax(),
        )

    # And this part defines the way they are connected to each other
    # (In reality, it is our forward pass)
    def forward(self, x):
        # ----- Conv -----
        out = self.conv_fw(x)

        # ----- Fully Connected -----
        # Imaginary layer
        # out = out.view(-1, self.fc_in_size)
        out = flatten(out, 1)

        out = self.fc_fw(out)

        return out


def get_model(
    model_type: Literal["MyNet", "LeNet5", "VGG11", "VGG13", "VGG16"],
    act_fn,
    dropout_prob,
    img_size,
):
    print(f'Loading model... {model_type}')
    model = None
    act_fn = ACT_FN_DICT[act_fn]

    set_seed()
    if model_type == "MyNet":
        model = MyNet(act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size)

    elif model_type == 'LeNet5':
        model = LeNet5(act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size)

    elif model_type == "VGG11":
        model = VGG11(act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size)

    elif model_type == "VGG13":
        model = VGG13(act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size)

    elif model_type == 'VGG16':
        model = VGG16(act_fn=act_fn, dropout_prob=dropout_prob, img_size=img_size)

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model = model.to(DEVICE)

    return model


def log_model_info(model, img_size, neptune_run: neptune.Run):
    model_info = summary(
        model,
        input_size=(
            1,
            N_CHANNEL,
            img_size,
            img_size,
        ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
    neptune_run["model/type"] = str(type(model))
    neptune_run["model/summary"].upload(File.from_content(str(model_info)))


def get_optimizer(
    network: nn.Module,
    type: Literal["SGD", "Adam"] = "SGD",
    lr: float = 0.01,
    momentum: float = 0.5,
):
    if type == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)

    elif type == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=lr)

    return optimizer


def get_loss_fn(type: Literal["nll", "cross_entropy"] = "cross_entropy"):
    return nn.CrossEntropyLoss()
