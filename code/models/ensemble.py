from typing import Literal
from lightgbm import LGBMClassifier
import numpy as np
from tensorflow import keras

import sys
import os
import random

sys.path.append('./')
sys.path.append('../')

from attention_model import Attention
from autoencoder_tree import AutoencoderTree
from vanilla_cnn import train_vanilla_cnn
from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import upsample

UPSAMPLE = False
SEED = 2137


class Ensemble:
    """Ensemble method"""

    def __init__(self, dataset: Literal['mit', 'ptb'],
                 method: Literal['probs', 'feats']):
        self.dataset = dataset
        if method == "probs":
            self.featuriser = self.probabilities
        else:
            self.featuriser = self.features
        self.method = method
        self.load_data(dataset)

    def init_models(self):

        self.attention_model = Attention(dataset=self.dataset)
        self.attention_model.train(load_model=True)

        self.autoencoder_tree_model = AutoencoderTree(dataset='ptb',
                                                      train_ae_on='full')
        self.autoencoder_tree_model.train()

        #self.vanilla_cnn_model, self.vanilla_cnn_trainer = train_vanilla_cnn(channels = [1, 20, 20, 40], kernel_size = 10,cnn_output_size = 187,dataset = "ptbdb")

    def probabilities(self, X):

        X_probs = np.concatenate(
            (self.attention_model.clf.predict(X),
             self.autoencoder_tree_model.clf.predict_proba(
                 (self.autoencoder_tree_model.featurizer.predict(
                     self.autoencoder_tree_model._pad(X))))),
            axis=1)
        print(X_probs.shape)
        return X_probs

    def features(self, X):

        ## attention model features
        attention_model_features = keras.Model(
            self.attention_model.clf.input,
            self.attention_model.clf.layers[-3].output)

        X_feats = np.concatenate(
            (attention_model_features.predict(X),
             self.autoencoder_tree_model.featurizer.predict(
                 self.autoencoder_tree_model._pad(X))),
            axis=1)

        return X_feats

    def train(self):
        print("Training individual models...")
        self.init_models()
        X_probs = self.featuriser(self.X_train)
        print(X_probs.shape)

        print("Training ensemble classifier...")
        if self.method == 'probs':
            self.clf = LGBMClassifier(n_estimators=30, seed=SEED)
        else:
            self.clf = LGBMClassifier(n_estimators=200, seed=SEED)
        self.clf.fit(X_probs, self.y_train)

    def predict(self):
        X_test = self.featuriser(self.X_test)
        self.y_pred = self.clf.predict(X_test)

    def load_data(self, dataset):
        if dataset == 'mit':
            X_train, y_train, X_test, y_test = load_arrhythmia_dataset()
        elif dataset == 'ptb':
            X_train, y_train, X_test, y_test = load_PTB_dataset()
        else:
            raise ValueError(dataset, 'is not a valid dataset')

        if UPSAMPLE:
            X_train, y_train = upsample(X_train, y_train)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(SEED)
        tf.random.set_seed(SEED)
        tf.keras.initializers.glorot_normal(SEED)
        np.random.seed(SEED)
        random.seed(SEED)


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix, accuracy_score

    model = Ensemble(dataset='ptb', method='feats')
    model.train()
    model.predict()
    cm = confusion_matrix(model.y_test, model.y_pred)
    print(cm)
    accuracy = accuracy_score(model.y_test, model.y_pred)
    print('Accuracy:', accuracy)