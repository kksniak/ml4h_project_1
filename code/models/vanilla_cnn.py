import imp
import torch
from torch import nn
from torch import zeros
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import get_predictions, prepare_datasets
import pathlib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split


class vanillaCNN(pl.LightningModule):
    def __init__(
        self, channels: list[int], kernel_size: int, output_size: int, no_classes: int
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


def train_vanilla_cnn(
    channels: list[int],
    kernel_size: int,
    cnn_output_size: int,
    dataset: str,
    max_epochs: int = 15,
) -> tuple[pl.LightningModule, pl.Trainer]:

    if dataset == "mithb":
        x, y, x_test, y_test = load_arrhythmia_dataset()
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            x, y, x_test, y_test, True
        )
        model = vanillaCNN(channels, kernel_size, channels[-1] * cnn_output_size, 5)
    elif dataset == "ptbdb":
        x, y, x_test, y_test = load_PTB_dataset()
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            x, y, x_test, y_test, True, y_dtype=torch.float
        )
        model = vanillaCNN(channels, kernel_size, channels[-1] * cnn_output_size, 2)
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


def get_preds(model, trainer, X: np.ndarray) -> np.ndarray:
    datset = TensorDataset(torch.tensor(X, dtype=torch.float))
    loader = DataLoader(dataset=datset)
    preds = get_predictions(model, loader, trainer)

    return np.array(preds)
