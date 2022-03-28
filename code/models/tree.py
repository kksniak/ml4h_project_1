from typing import Literal
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import upsample, get_debug_data

DEBUG = False
UPSAMPLE = True
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 3
EARLY_STOPPING_PATIENCE = 2
FILTER_SIZE = 3
SEED = 1337


class Tree:
    """Extremely Randomized Trees model for heartbeat classification.
    
    Attributes:
        dataset: Literal specifying which dataset to use.
    """

    def __init__(self, dataset: Literal['mithb', 'ptbdb']):
        self.load_data(dataset)
        self.clf = ExtraTreesClassifier(n_estimators=100)

    def load_data(self, dataset: Literal['mithb', 'ptbdb']):
        """Loads and splits the dataset.

        The train and test datasets are loaded from files and split
        into a train and a validation dataset.

        Global options:
            DEBUG: if set to true, a small subsample of the dataset
                is used to expediate training.
            UPSAMPLE: if set to true, underrepresented classes are
                upsampled to match the sample size of the majority
                class.

        Args:
            dataset: Literal specifying which dataset to load.

        Raises:
            ValueError: Raised if an invalid dataset argument is
                passed.
        """
        if dataset == 'mit':
            X, y, X_test, y_test = load_arrhythmia_dataset()
        elif dataset == 'ptb':
            X, y, X_test, y_test = load_PTB_dataset()
        else:
            raise ValueError(dataset, 'is not a valid dataset')

        X_test = X_test[:, :, 0]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, stratify=y)

        if DEBUG:
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_debug_data(
                X_train, y_train, X_valid, y_valid, X_test, y_test)

        if UPSAMPLE:
            X_train, y_train = upsample(X_train, y_train)
            X_train = X_train[:, :, 0]

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        print('Fitting classifier...')
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sn
    import matplotlib.pyplot as plt
    from evaluation import evaluate

    np.random.seed(SEED)

    model = Tree(dataset='mithb')
    model.train()
    model.predict()

    evaluate('Tree', model.y_pred_proba, model.y_test, save_results=False)
