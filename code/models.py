import torch
from torch import nn
from torch import zeros
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
    def __init__(self, no_hidden: int, no_classes: int, num_layers: int = 1) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=1, hidden_size=no_hidden, batch_first=True, num_layers=num_layers
        )
        self.num_layers = num_layers
        if no_classes == 2:
            self.fc = nn.Linear(no_hidden * num_layers, 1)

        else:
            self.fc = nn.Linear(no_hidden * num_layers, no_classes)

        self.no_classes = no_classes
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # print(x.shape)
        output, hn = self.rnn(x)
        D_num, batch_size, hidden_size = hn.shape
        x = self.fc(hn.view(1, batch_size, hidden_size * self.num_layers))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print("Batch: ", x.shape)
        output, hn = self.rnn(x)
        D_num, batch_size, hidden_size = hn.shape
        pred = self.fc(hn.view(1, batch_size, hidden_size * self.num_layers))
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
            y_hat = y_hat.squeeze()
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

        pred.squeeze_()
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return optimizer

        # h0 = (torch.autograd.Variable(zeros(1, 64, hidden_size)),
        #       torch.autograd.Variable(zeros(1, 64, hidden_size)))


class LSTM(pl.LightningModule):
    def __init__(self, hidden_size: int, num_classes: int, num_layers: int) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size

        self.num_classes = num_classes
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, X):
        # h0 = torch.zeros(1, 64, self.hidden_size)
        # c0 = torch.zeros(1, 64, self.hidden_size)
        # output, _ = self.lstm(X, (h0, c0))
        output, _ = self.lstm(X)
        # (batch_size, sequence_length, hidden_size)
        y_hat = self.fc(output[:, -1, :])
        return y_hat

    def training_step(self, batch, batch_idx):
        X, y = batch
        output, _ = self.lstm(X)
        pred = self.fc(output[:, -1, :])
        pred = pred.squeeze()
        loss = F.cross_entropy(pred, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)

        y_hat = y_hat.squeeze()
        val_loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", val_loss)
        self.accuracy(y_hat, y.long())
        self.log("val_acc", self.accuracy)

    def predict_step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X)
        pred = pred.squeeze()
        return pred

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.7)
        return optimizer

    def predict_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
