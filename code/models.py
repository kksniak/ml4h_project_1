from turtle import forward
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class vanillaCNN(pl.LightningModule):
    def __init__(
        self, channels: list[int], kernel_size: int, output_size, no_classes: int
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()

        for i in range(len(channels) - 1):
            self.net.add_module(
                f"cnn{i}",
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size),
            )
            self.net.add_module(f"act{i}", nn.ReLU())

        self.no_classes = no_classes

        if no_classes == 2:
            self.fc = nn.Linear(output_size, 1)

        else:
            self.fc = nn.Linear(output_size, no_classes)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x.unsqueeze_(1)
        x = self.net(x)

        dims = x.shape
        x = x.view(-1, dims[1] * dims[2])
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.unsqueeze_(1)

        output_cnn = self.net(x)

        dims = output_cnn.shape
        output_cnn = output_cnn.view(-1, dims[1] * dims[2])
        output = self.fc(output_cnn)

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


class vanillaRNN(pl.LightningModule):
    def __init__(self, no_hidden: int, no_classes: int) -> None:
        super().__init__()

        self.rnn = nn.RNN(input_size=1, hidden_size=no_hidden, batch_first=True)
        if no_classes == 2:
            self.fc = nn.Linear(no_hidden, 1)

        else:
            self.fc = nn.Linear(no_hidden, no_classes)

        self.no_classes = no_classes
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        output, hn = self.rnn(x)
        x = self.fc(hn)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, hn = self.rnn(x)
        pred = self.fc(hn)
        if self.no_classes == 2:
            pred = pred.squeeze()
            loss = F.binary_cross_entropy(torch.sigmoid(pred), y)
        else:
            pred.squeeze_(0)
            # print("pred size train: ", pred.shape)
            loss = F.cross_entropy(pred, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print("X ", x.shape)
        y_hat = self.forward(x)

        if self.no_classes == 2:
            # y_hat = y_hat.squeeze()
            y_hat = torch.sigmoid(y_hat)
            val_loss = F.binary_cross_entropy(y_hat, y)
        else:
            y_hat.squeeze_(0)
            # print("pred size vale: ", y_hat.shape)

            val_loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", val_loss)
        self.accuracy(y_hat, y.long())
        self.log("val_acc", self.accuracy)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        if self.no_classes == 2:
            pred = torch.sigmoid(pred)
        else:
            pred.squeeze_()
        return pred

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.7)
        return optimizer
