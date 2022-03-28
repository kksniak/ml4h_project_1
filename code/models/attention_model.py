from typing import Literal
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    MultiHeadAttention,
    Conv1D,
    Dropout,
    GlobalMaxPooling1D,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import sys
import os
import random

sys.path.append("./")
sys.path.append("../")

from datasets import load_PTB_dataset, load_arrhythmia_dataset
from utils import upsample

VALIDATION_SPLIT = 0.1
BATCH_SIZE = 64
UPSAMPLE = False
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
SEED = 2137


class Attention:
    """Attention model"""

    def __init__(self, dataset: Literal["mithb", "ptbdb"]):
        self.dataset = dataset
        self.load_data(dataset)

    def init_attention_model(
        self,
        input_shape,
        n_classes,
        n_encoders=2,
        key_dim=8,
        num_heads=8,
        feedforward_layers=[64],
    ):
        """Method defines an attention model

        Args:
            input_shape (np.array): Shape of input data.
            n_classes (int): Number of different classes for softmax.
            n_encoders (int, optional): Number of transfer encoders in model, in sequantiall manner. Defaults to 2.
            key_dim (int, optional): Size of each attention head for query and key. Defaults to 8.
            num_heads (int, optional):  Number of attention heads. Defaults to 8.
            feedforward_layers (list, optional): Filters in convolutional layers in each encoder (the more filters the more layers). Defaults to [64].
        """

        inputs = keras.Input(shape=input_shape)
        x = inputs

        for _ in range(n_encoders):

            res_1 = x
            x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=0.1)(
                x, x
            )
            res_2 = x + res_1
            x = res_2

            for filter in feedforward_layers:
                x = Conv1D(filters=filter, kernel_size=1, activation="relu")(x)
                x = Dropout(0.1)(x)
                x = Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(
                    x
                )
                x = x + res_2

        x = GlobalMaxPooling1D(data_format="channels_first")(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(n_classes, activation="softmax")(x)

        self.clf = keras.Model(inputs, outputs)

    def train(self, load_model: boolean):
        """Needs to be called on class to train models.

        Args:
            load_model (boolean): If set to true, instead of training model from scratch, loads pretrained models
        """
        self.set_seeds()

        if load_model:
            print("Loading attention model...")
            self.load_model(self.dataset)
            return

        n_classes = len(np.unique(self.y_train))
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        )
        redonplat = ReduceLROnPlateau(
            monitor="val_acc", factor=0.5, mode="max", patience=3, verbose=2
        )
        callbacks = [early_stopping, redonplat]

        print("Fitting attention model...")
        self.init_attention_model(
            self.X_train.shape[1:],
            n_classes,
            n_encoders=2,
            key_dim=8,
            num_heads=8,
            feedforward_layers=[64],
        )

        self.clf.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["acc"],
        )

        self.clf.fit(
            self.X_train,
            self.y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
        )

    def transfer_learning_method_1(self):
        """Trains models using transfer learning with frozen layers (apart from last 3 dense layers).
        """
        self.set_seeds()

        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        )
        redonplat = ReduceLROnPlateau(
            monitor="val_acc", factor=0.5, mode="max", patience=3, verbose=2
        )
        callbacks = [early_stopping, redonplat]

        print("Fitting PTB dataset into attention model (training whole model)...")
        pretrained_model = keras.models.load_model(
            "models/attention_model_checkpoints/arythmia_checkpoint"
        )

        # replace output layer
        self.clf = keras.Model(
            inputs=pretrained_model.input,
            outputs=Dense(2, activation="softmax", name="dense_3")(
                pretrained_model.layers[-2].output
            ),
        )

        # freeze layers (apart from feedforwark network)
        for layer in self.clf.layers[:-4]:
            layer.trainable = False

        self.clf.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["acc"],
        )
        self.clf.fit(
            self.X_train,
            self.y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
        )

    def transfer_learning_method_2(self):
        """Transfer Learning without any frozen layers."""
        self.set_seeds()

        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        )
        redonplat = ReduceLROnPlateau(
            monitor="val_acc", factor=0.5, mode="max", patience=3, verbose=2
        )
        callbacks = [early_stopping, redonplat]

        print("Fitting PTB dataset into attention model (training whole model)...")
        pretrained_model = keras.models.load_model(
            "models/attention_model_checkpoints/arythmia_checkpoint"
        )

        # replace output layer
        self.clf = keras.Model(
            inputs=pretrained_model.input,
            outputs=Dense(2, activation="softmax", name="dense_3")(
                pretrained_model.layers[-2].output
            ),
        )

        self.clf.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["acc"],
        )
        self.clf.fit(
            self.X_train,
            self.y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
        )

    def load_model(self, dataset: Literal["mithb", "ptbdb"]):
        """Loads pretrained models"""
        if dataset == "mithb":
            self.clf = keras.models.load_model(
                "models/attention_model_checkpoints/arythmia_checkpoint"
            )
        else:
            self.clf = keras.models.load_model(
                "models/attention_model_checkpoints/ptb_checkpoint"
            )

    def predict(self):
        """Generate predictions on test set"""
        self.y_pred = self.clf.predict(self.X_test)

    def load_data(self, dataset):
        """Loads data"""
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
        """Set seeds for repreducability"""
        os.environ["PYTHONHASHSEED"] = str(SEED)
        tf.random.set_seed(SEED)
        tf.keras.initializers.glorot_normal(SEED)
        np.random.seed(SEED)
        random.seed(SEED)


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix, accuracy_score

    base_model = Attention(dataset="ptbdb")
    base_model.train(load_model=True)
    base_model.predict()
    cm = confusion_matrix(base_model.y_test, np.argmax(base_model.y_pred, axis=1))
    print(cm)
    accuracy = accuracy_score(base_model.y_test, np.argmax(base_model.y_pred, axis=1))
    print("Accuracy:", accuracy)

    transfer_model = Attention(dataset="ptbdb")
    transfer_model.transfer_learning_method_1()
    transfer_model.predict()
    cm = confusion_matrix(
        transfer_model.y_test, np.argmax(transfer_model.y_pred, axis=1)
    )
    print(cm)
    accuracy = accuracy_score(
        transfer_model.y_test, np.argmax(transfer_model.y_pred, axis=1)
    )
    print("Accuracy:", accuracy)

    transfer_model_2 = Attention(dataset="ptbdb")
    transfer_model_2.transfer_learning_method_2()
    transfer_model_2.predict()
    cm = confusion_matrix(
        transfer_model_2.y_test, np.argmax(transfer_model_2.y_pred, axis=1)
    )
    print(cm)
    accuracy = accuracy_score(
        transfer_model_2.y_test, np.argmax(transfer_model_2.y_pred, axis=1)
    )
    print("Accuracy:", accuracy)

