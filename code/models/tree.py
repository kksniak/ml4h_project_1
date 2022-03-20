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
    """Autoencoder featurization model combined with tree model"""

    def __init__(self, dataset: Literal['mit', 'ptb']):
        self.load_data(dataset)
        self.clf = ExtraTreesClassifier(n_estimators=100)

    def load_data(self, dataset):
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

    np.random.seed(SEED)

    model = Tree(dataset='mit')
    model.train()
    model.predict()

    cm = confusion_matrix(model.y_test, model.y_pred)
    sn.heatmap(cm, annot=True, fmt='g')
    plt.show()

    accuracy = accuracy_score(model.y_test, model.y_pred)
    print('Accuracy:', accuracy)
