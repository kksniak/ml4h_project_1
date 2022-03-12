import imp
from typing import Tuple
import pandas as pd
import numpy as np
import pathlib
from typing import Tuple
from sklearn.model_selection import train_test_split


def load_arythmia_dataset() -> Tuple[np.array, np.array, np.array, np.array]:
    data_dir = pathlib.Path(__file__).parents[1].joinpath("data")
    df_train = pd.read_csv(
        pathlib.Path(data_dir).joinpath("mitbih_train.csv"), header=None
    )
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(
        pathlib.Path(data_dir).joinpath("mitbih_test.csv"), header=None
    )

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return X, Y, X_test, Y_test


def load_PTB_dataset() -> Tuple[np.array, np.array, np.array, np.array]:
    data_dir = pathlib.Path(__file__).parents[1].joinpath("data")
    df_1 = pd.read_csv(pathlib.Path(data_dir).joinpath("ptbdb_normal.csv"), header=None)
    df_2 = pd.read_csv(
        pathlib.Path(data_dir).joinpath("ptbdb_abnormal.csv"), header=None
    )
    df = pd.concat([df_1, df_2])

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1337, stratify=df[187]
    )

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return X, Y, X_test, Y_test
