from typing import Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.utils import resample
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sn
import torch.nn.functional as F
import os
import tensorflow as tf
import random
import torch


def CNN_output_shape(
    input_size: int = 188,
    dilation: int = 1,
    kernel_size: int = 3,
    padding: int = 0,
    stride: int = 1,
) -> int:
    """Computes output shape of the data, based on parameters of 1D CNN layer"""
    output = int(
        ((input_size + 2 * padding - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )

    return output


def get_predictions(
    model: pl.LightningModule, data_loader: DataLoader, trainer: pl.Trainer
) -> np.ndarray:
    """Returns prediction as numpy array using privided model, trainer and DataLoader"""
    preds = trainer.predict(model, data_loader)
    test_preds = []
    for pred in preds:

        test_preds.append(pred.numpy())

    test_preds = np.concatenate(test_preds)

    return test_preds


def get_preds_from_numpy(
    model: pl.LightningModule, trainer: pl.Trainer, X: np.ndarray, softmax=True
) -> np.ndarray:
    """Generates predictions from data as numpy array and returns it as numpy array using provided PyTorch Lightning model and trainer"""
    datset = TensorDataset(torch.tensor(X, dtype=torch.float).squeeze())
    loader = DataLoader(dataset=datset, batch_size=64)
    preds = get_predictions(model, loader, trainer)

    if len(preds.shape) == 1:
        preds = np.expand_dims(preds, axis=1)

    if softmax:
        if preds.shape[1] == 1:
            torch.sigmoid(torch.tensor(preds))
        else:
            preds = F.softmax(torch.tensor(preds), dim=1)

    return np.array(preds)


def prepare_datasets(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    squeeze: bool = True,
    y_dtype=torch.long,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Creates train, validation and test dataset as TensorDataset from provided data in numpy arrays """

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


def upsample(X, y):
    """Upsample minority classes to the size of the majority class"""

    df = pd.DataFrame(X[:, :, 0])
    df["y"] = y

    # Create one dataframe for each class
    dfs = [df[df["y"] == i] for i in range(np.max(y) + 1)]

    # Use the size of the biggest class as the target size when resampling
    sample_size = np.max(df["y"].value_counts())

    # Majority class shall remain unchanged
    resampled_dfs = [dfs[0]]

    for i in range(1, len(dfs)):
        df = dfs[i]
        resampled_df = resample(df, n_samples=sample_size, replace=True)
        resampled_dfs.append(resampled_df)

    df = pd.concat(resampled_dfs)

    y_upsampled = df["y"]
    X_upsampled = df.drop(columns=["y"])

    X_upsampled = np.array(X_upsampled.values)[..., np.newaxis]
    y_upsampled = np.array(y_upsampled.values).astype(np.int8)

    return X_upsampled, y_upsampled


def get_debug_data(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    sample_size=100,
    validation_split=0.1,
    test_split=0.2,
):

    X_train, y_train = resample(
        X_train,
        y_train,
        replace=False,
        n_samples=sample_size * (1 - validation_split),
        stratify=y_train,
    )
    X_valid, y_valid = resample(
        X_valid,
        y_valid,
        replace=False,
        n_samples=sample_size * validation_split,
        stratify=y_valid,
    )
    X_test, y_test = resample(
        X_test,
        y_test,
        replace=False,
        n_samples=test_split * sample_size,
        stratify=y_test,
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    tf.keras.initializers.glorot_normal(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
