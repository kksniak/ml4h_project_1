from typing import Literal
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from xgboost import XGBClassifier
import numpy as np
from tensorflow import keras

import sys
import os
import random

sys.path.append("./")
sys.path.append("../")

from models.attention_model import Attention
from models.autoencoder_tree import AutoencoderTree
from models.vanilla_cnn import train_vanilla_cnn, get_cnn_outputs
from models.resnet1d import train_resnet
from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import upsample, get_preds_from_numpy

UPSAMPLE = False
SEED = 2137


class Ensemble:
    """Ensemble method"""

    def __init__(self, dataset: Literal["mithb", "ptbdb"],
                 method: Literal["probs", "feats"]):
        """Class initialisation

        Args:
            dataset (Literal[&quot;mithb&quot;, &quot;ptbdb&quot;]): Dataset for training and testing
            method (Literal[&quot;probs&quot;, &quot;feats&quot;]): Ensemble method used. Probs corresponds to ensmeble on softmax layers, while feats on hidden layers before Dense classifiers.
        """
        self.dataset = dataset
        if method == "probs":
            self.featuriser = self.probabilities
        else:
            self.featuriser = self.features
        self.method = method
        self.load_data(dataset)

    def init_models(self, models: dict = {}):
        """method training/initialising sub-models used for ensemble

        Args:
            models (dict, optional): Dictionary of already trained models (restricted to categories attention, autoencoder, CNN, and ResNet models). Defaults to {}.
        """

        if "attention" in models.keys():
            self.attention_model = models["attention"]
        else:
            self.attention_model = Attention(dataset=self.dataset)
            self.attention_model.train(load_model=True)

        if "autoencoder" in models.keys():
            self.autoencoder_tree_model = models["autoencoder"]
        else:
            self.autoencoder_tree_model = AutoencoderTree(dataset=self.dataset,
                                                          train_ae_on="full")
            self.autoencoder_tree_model.train()

        if "cnn" in models.keys():
            self.vanilla_cnn_model, self.vanilla_cnn_trainer = models["cnn"]
        else:
            self.vanilla_cnn_model, self.vanilla_cnn_trainer = train_vanilla_cnn(
                channels=[1, 20, 20, 40],
                kernel_size=10,
                cnn_output_size=160,
                dataset=self.dataset)

        if "resnet" in models.keys():
            self.resnet_model, self.resnet_trainer = models["resnet"]
        else:
            self.resnet_model, self.resnet_trainer = train_resnet(
                [10, 20, 20, 40], dataset=self.dataset, max_epochs=15)

    def probabilities(self, X):
        """Ensemble method using softmax layers/probabilities for each function as features

        Args:
            X (np.arrray): either train or test set

        Returns:
            np.arrays: array of features for each sample
        """
        X_probs = np.concatenate(
            (
                self.attention_model.clf.predict(X),
                self.autoencoder_tree_model.clf.predict_proba(
                    (self.autoencoder_tree_model.featurizer.predict(
                        self.autoencoder_tree_model._pad(X)))),
                get_preds_from_numpy(self.vanilla_cnn_model,
                                     self.vanilla_cnn_trainer, X),
                get_preds_from_numpy(self.resnet_model, self.resnet_trainer, X),
            ),
            axis=1,
        )
        print("X_probs_shape", X_probs.shape)
        return X_probs

    def features(self, X):
        """Ensemble method using hidden layers/latent space representation found before classifier in each mode.

        Args:
            X (np.arrray): either train or test set

        Returns:
            np.arrays: array of features for each sample
        """

        attention_model_features = keras.Model(
            self.attention_model.clf.input,
            self.attention_model.clf.layers[-3].output)

        X_feats = np.concatenate(
            (
                attention_model_features.predict(X),
                self.autoencoder_tree_model.featurizer.predict(
                    self.autoencoder_tree_model._pad(X)),
                get_cnn_outputs(self.vanilla_cnn_model, X),
                get_preds_from_numpy(
                    self.resnet_model, self.resnet_trainer, X, softmax=False),
            ),
            axis=1,
        )
        print("X_probs_feats", X_feats.shape)
        return X_feats

    def train(self, models: dict = {}):
        """Trains ensmble classifier on samples, and performs feature selection if needed"""

        print("Training individual models...")
        self.init_models(models)
        self.X_train = self.featuriser(self.X_train)

        if self.dataset == 'mithb' and self.method == "feats":
            print("Feature Selection...")
            self.selector_1 = VarianceThreshold()
            self.X_train = self.selector_1.fit_transform(
                self.X_train, self.y_train)
            self.selector_2 = SelectKBest(f_classif, k=200)
            self.X_train = self.selector_2.fit_transform(
                self.X_train, self.y_train)

        print(self.X_train.shape)

        print("Training ensemble classifier...")
        if self.method == "probs":
            self.clf = ExtraTreesClassifier(n_estimators=100)
        else:
            self.clf = XGBClassifier(seed=SEED, verbosity=1)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        """Prediction and feature selection if needed"""
        self.X_test = self.featuriser(self.X_test)
        if self.dataset == 'mithb' and self.method == "feats":
            self.X_test = self.selector_1.transform(self.X_test)
            self.X_test = self.selector_2.transform(self.X_test)

        self.y_pred = self.clf.predict(self.X_test)
        self.y_pred_proba = self.clf.predict_proba(self.X_test)

    def load_data(self, dataset):
        """Loads datasets"""
        if dataset == "mithb":
            X_train, y_train, X_test, y_test = load_arrhythmia_dataset()
        elif dataset == "ptbdb":
            X_train, y_train, X_test, y_test = load_PTB_dataset()
        else:
            raise ValueError(dataset, "is not a valid dataset")

        if UPSAMPLE:
            X_train, y_train = upsample(X_train, y_train)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_seeds(self):
        """Set seeds for reproducability"""
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        random.seed(SEED)


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix, accuracy_score

    model = Ensemble(dataset="ptbdb", method="probs")

    model.train()
    model.predict()
    cm = confusion_matrix(model.y_test, model.y_pred)
    print(cm)
    accuracy = accuracy_score(model.y_test, model.y_pred)
    print("Accuracy:", accuracy)
