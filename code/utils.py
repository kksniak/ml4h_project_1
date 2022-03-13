import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np


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
