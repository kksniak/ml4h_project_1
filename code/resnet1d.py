from turtle import forward
from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        change_no_channels: bool = False,
    ) -> None:
        super().__init__()

        norm_layer = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4)
        self.bn2 = norm_layer(out_channels)

        self.conv_for_x = nn.Conv1d(in_channels, out_channels, 1)
        self.change_no_channels = change_no_channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # print("x: ", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.change_no_channels:
            identity = self.conv_for_x(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    def __init__(self, channels: List[int], no_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels[0], 12)
        self.bn1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(channels[0], channels[1],change_no_channels=not(channels[0]==channels[1]))
        self.max_pool = nn.MaxPool1d(3, 2)

        self.block2 = BasicBlock(channels[1], channels[2], change_no_channels=True)
        self.block3 = BasicBlock(channels[2], channels[3], change_no_channels=True)
        output_size = self._cnn_pass(torch.rand(16, 1, 187))

        if no_classes == 2:
            self.fc = nn.Linear(output_size * channels[3], 1)
        else:
            self.fc = nn.Linear(output_size * channels[3], no_classes)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        self.relu(x)
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.block2(x)
        x = self.block3(x)
        # print("CNN out: ", x.shape)
        dims = x.shape
        x = self.fc(x.view(dims[0], dims[1] * dims[2]))

        return x

    def _cnn_pass(self, x: Tensor):

        x = self.conv1(x)
        x = self.bn1(x)
        self.relu(x)
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.shape[2]
