import imp
from typing import Tuple
import pandas as pd
import numpy as np
import pathlib
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def get_subset(X, Y, X_test, Y_test, n_samples):
    X, Y = resample(
        X,
        Y,
        replace=False,
        n_samples=n_samples,
        stratify=Y,
    )
    X_test, Y_test = resample(
        X_test,
        Y_test,
        replace=False,
        n_samples=n_samples,
        stratify=Y_test,
    )

    return X, Y, X_test, Y_test


def load_arrhythmia_dataset(
        n_samples=-1) -> Tuple[np.array, np.array, np.array, np.array]:
    """Loads MIT dataset as specified in the assignment

    Args:
        n_samples: Optional argument specifying the size of a random
            stratified subset to be loaded. Used for debugging
            purposes.

    Returns:
        Loaded dataset as numpy arrays
    """
    data_dir = pathlib.Path(__file__).parents[1].joinpath("data")
    df_train = pd.read_csv(pathlib.Path(data_dir).joinpath("mitbih_train.csv"),
                           header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(pathlib.Path(data_dir).joinpath("mitbih_test.csv"),
                          header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    if n_samples > 0:
        X, Y, X_test, Y_test = get_subset(X, Y, X_test, Y_test, n_samples)

    return X, Y, X_test, Y_test


def load_PTB_dataset(
        n_samples=-1) -> Tuple[np.array, np.array, np.array, np.array]:
    """Loads PTB dataset as specified in the assignment

    Args:
        n_samples: Optional argument specifying the size of a random
            stratified subset to be loaded. Used for debugging
            purposes.

    Returns:
        Loaded dataset as numpy arrays
    """
    data_dir = pathlib.Path(__file__).parents[1].joinpath("data")
    df_1 = pd.read_csv(pathlib.Path(data_dir).joinpath("ptbdb_normal.csv"),
                       header=None)
    df_2 = pd.read_csv(pathlib.Path(data_dir).joinpath("ptbdb_abnormal.csv"),
                       header=None)
    df = pd.concat([df_1, df_2])

    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=1337,
                                         stratify=df[187])

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    if n_samples > 0:
        X, Y, X_test, Y_test = get_subset(X, Y, X_test, Y_test, n_samples)

    return X, Y, X_test, Y_test
