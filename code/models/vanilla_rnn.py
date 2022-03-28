import torch
from torch import nn
from torch import zeros
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import prepare_datasets
from config import USE_GPU


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
        output, hn = self.rnn(x)
        D_num, batch_size, hidden_size = hn.shape
        x = self.fc(hn.view(1, batch_size, hidden_size * self.num_layers))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, hn = self.rnn(x)
        D_num, batch_size, hidden_size = hn.shape
        pred = self.fc(hn.view(1, batch_size, hidden_size * self.num_layers))
        if self.no_classes == 2:
            pred = pred.squeeze()
            loss = F.binary_cross_entropy(torch.sigmoid(pred), y)
        else:
            pred.squeeze_(0)
            loss = F.cross_entropy(pred, y)

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
            y_hat.squeeze_(0)

            val_loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", val_loss)
        self.accuracy(y_hat, y.long())
        self.log("val_acc", self.accuracy)

    def predict_step(self, batch, batch_idx):
        if len(batch) == 1:
            x = batch[0]
        else:
            x, y = batch

        if len(x.shape) < 3:
            x.unsqueeze_(2)

        pred = self.forward(x)
        if self.no_classes == 2:
            pred = torch.sigmoid(pred)

        pred.squeeze_()
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer


def train_vanilla_rnn(
    no_hidden: int, dataset: str, num_layers: int = 1, max_epochs: int = 15
) -> tuple[pl.LightningModule, pl.Trainer]:
    """_summary_

    Args:
        no_hidden: Size of hidden state
        dataset: name of dataset the model will be traind on, either "mithb" or "ptbdb"
        num_layers: Number of RNN layers. Defaults to 1.
        max_epochs: Maximum number of training epochs. Defaults to 15.

    Raises:
        ValueError: If provided incorrect dataset name.

    Returns:
        Trained model and trainer.
    """
    if dataset == "mithb":
        x, y, x_test, y_test = load_arrhythmia_dataset()
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            x, y, x_test, y_test, False
        )
        model = vanillaRNN(no_hidden, 5, num_layers)
    elif dataset == "ptbdb":
        x, y, x_test, y_test = load_PTB_dataset()
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            x, y, x_test, y_test, False, torch.float
        )
        model = vanillaRNN(no_hidden, 2, num_layers)
    else:
        raise ValueError("Incorrect dataset!")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    trainer = trainer = pl.Trainer(
        gpus=USE_GPU,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        gradient_clip_val=0.5,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,
    )

    return model, trainer
