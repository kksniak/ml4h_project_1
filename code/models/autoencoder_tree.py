from typing import Literal
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

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


class AutoencoderTree:
    """Autoencoder featurization model combined with tree model"""

    def __init__(self, dataset: Literal['mit', 'ptb']):
        self.init_autoencoder()
        self.load_data(dataset)
        self.clf = ExtraTreesClassifier(n_estimators=100)

    def init_autoencoder(self):
        input_shape = (188, 1)

        self.autoencoder = Sequential([
            Conv1D(2,
                   FILTER_SIZE,
                   strides=2,
                   activation='relu',
                   padding='same',
                   input_shape=input_shape),
            Conv1D(4, FILTER_SIZE, strides=2, activation='relu',
                   padding='same'),
            Conv1DTranspose(4, FILTER_SIZE, strides=2, padding='same'),
            Conv1DTranspose(2, FILTER_SIZE, strides=2, padding='same'),
            Conv1DTranspose(1, 1, strides=1, padding='same')
        ])

        self.autoencoder.compile(loss='mse',
                                 optimizer=Adam(0.001),
                                 metrics=['mse'])

    def load_data(self, dataset):
        if dataset == 'mit':
            X, y, X_test, y_test = load_arrhythmia_dataset()
        elif dataset == 'ptb':
            X, y, X_test, y_test = load_PTB_dataset()
        else:
            raise ValueError(dataset, 'is not a valid dataset')
        X = np.append(X, np.zeros((X.shape[0], 1, 1)), axis=1)
        X_test = np.append(X_test, np.zeros((X_test.shape[0], 1, 1)), axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, stratify=y)

        if DEBUG:
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_debug_data(
                X_train, y_train, X_valid, y_valid, X_test, y_test)

        if UPSAMPLE:
            X_train, y_train = upsample(X_train, y_train)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        print('Fitting autoencoder...')
        self.autoencoder.fit(self.X_train,
                             self.X_train,
                             batch_size=BATCH_SIZE,
                             validation_data=(self.X_valid, self.X_valid),
                             shuffle=True,
                             epochs=EPOCHS,
                             callbacks=[
                                 tf.keras.callbacks.EarlyStopping(
                                     monitor='val_loss',
                                     patience=EARLY_STOPPING_PATIENCE)
                             ])

        print('Generating training features...')
        flat = Flatten()(self.autoencoder.layers[2].output)
        self.featurizer = Model(self.autoencoder.input, flat)

        X_train = self.featurizer.predict(self.X_train)

        print('Fitting classifier...')
        self.clf.fit(X_train, self.y_train)

    def predict(self):
        X_test = self.featurizer.predict(self.X_test)
        self.y_pred = self.clf.predict(X_test)


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sn
    import matplotlib.pyplot as plt

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = AutoencoderTree(dataset='mit')
    model.train()
    model.predict()

    cm = confusion_matrix(model.y_test, model.y_pred)
    sn.heatmap(cm, annot=True, fmt='g')
    plt.show()

    accuracy = accuracy_score(model.y_test, model.y_pred)
    print('Accuracy:', accuracy)
