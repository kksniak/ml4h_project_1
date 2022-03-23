from turtle import forward
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import prepare_datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import pathlib
import torchmetrics
import torch.nn.functional as F


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
        self.block1 = BasicBlock(
            channels[0],
            channels[1],
            change_no_channels=not (channels[0] == channels[1]),
        )
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


class ResNet(pl.LightningModule):
    def __init__(self, channels: list[int], no_classes: int) -> None:
        super().__init__()

        self.net = ResNet1d(channels, no_classes)
        self.no_classes = no_classes
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x.unsqueeze_(1)

        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.unsqueeze_(1)

        output = self.net(x)

        if self.no_classes == 2:
            output = output.squeeze()
            loss = F.binary_cross_entropy(torch.sigmoid(output), y)
        else:
            loss = F.cross_entropy(output, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        if self.no_classes == 2:
            y_hat = y_hat.squeeze()
            y_hat = torch.sigmoid(y_hat)
            val_loss = F.binary_cross_entropy(y_hat, y)
        else:
            val_loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", val_loss)
        self.accuracy(y_hat, y.long())
        self.log("val_acc", self.accuracy)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        if self.no_classes == 2:
            pred = torch.sigmoid(pred)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_resnet(channels: list[int], dataset: str, max_epochs: int = 15):
    if dataset == "mithb":
        x, y, x_test, y_test = load_arrhythmia_dataset()
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            x, y, x_test, y_test, True
        )
        model = ResNet(channels=channels, no_classes=5)
    elif dataset == "ptbdb":
        x, y, x_test, y_test = load_PTB_dataset()
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            x, y, x_test, y_test, True, y_dtype=torch.float
        )
        model = ResNet(channels=channels, no_classes=2)
    else:
        raise ValueError("Incorrect dataset!")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        default_root_dir=pathlib.Path(__file__).parents[1].joinpath("saved_models"),
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model, trainer


class ResNetTransferLearning(pl.LightningModule):
    def __init__(self, pretrained_model: ResNet) -> None:
        super().__init__()

        self.net = pretrained_model.net

        _, input_shape = self.net.fc.weight.shape

        self.net.fc = nn.Linear(input_shape, 1)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x.unsqueeze_(1)

        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.unsqueeze_(1)

        output = self.net(x)

        output = output.squeeze()
        loss = F.binary_cross_entropy(torch.sigmoid(output), y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        y_hat = y_hat.squeeze()
        y_hat = torch.sigmoid(y_hat)
        val_loss = F.binary_cross_entropy(y_hat, y)

        self.log("val_loss", val_loss)
        self.accuracy(y_hat, y.long())
        self.log("val_acc", self.accuracy)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        pred = torch.sigmoid(pred)

        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def perform_transfer_learning(max_epochs: int = 15):
    pretrained_model, _ = train_resnet([10, 20, 40, 40], "mithb", max_epochs)
    model = ResNetTransferLearning(pretrained_model=pretrained_model)
    x, y, x_test, y_test = load_PTB_dataset()
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        x, y, x_test, y_test, True, y_dtype=torch.float
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        default_root_dir=pathlib.Path(__file__).parents[1].joinpath("saved_models"),
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model, trainer

