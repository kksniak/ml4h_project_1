from typing import Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def CNN_output_shape(
    input_size: int = 188,
    dilation: int = 1,
    kernel_size: int = 3,
    padding: int = 0,
    stride: int = 1,
) -> int:
    output = int(
        ((input_size + 2 * padding - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )

    return output


def get_predictions(
    model: pl.LightningModule, data_loader: DataLoader, trainer: pl.Trainer
) -> np.ndarray:
    preds = trainer.predict(model, data_loader)
    test_preds = []
    for pred in preds:
        test_preds.append(pred.numpy())

    test_preds = np.concatenate(test_preds)
    # print(test_preds.shape)
    return test_preds


def prepare_datasets(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    squeeze: bool = True,
    y_dtype = torch.long
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    if squeeze:
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float).squeeze(),
            torch.tensor(y, dtype=y_dtype).squeeze(),
        )
        test_dataset = TensorDataset(
            torch.tensor(x_test, dtype=torch.float).squeeze(),
            torch.tensor(y_test, dtype=y_dtype).squeeze(),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float),
            torch.tensor(y, dtype=y_dtype).squeeze(),
        )

        test_dataset = TensorDataset(
            torch.tensor(x_test, dtype=torch.float),
            torch.tensor(y_test, dtype=y_dtype).squeeze(),
        )

    train_dataset, val_dataset = random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )

    return train_dataset, val_dataset, test_dataset

