from statistics import mode
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from tensorflow.keras import optimizers, losses, activations, models

from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
)
from keras.layers import (
    Dense,
    Input,
    Dropout,
    Convolution1D,
    MaxPool1D,
    GlobalMaxPool1D,
    GlobalAveragePooling1D,
    concatenate,
)


def get_PTBDB_baseline_model() -> models.Model:
    nclass = 1
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(
        16, kernel_size=5, activation=activations.relu, padding="valid"
    )(inp)
    img_1 = Convolution1D(
        16, kernel_size=5, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        256, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        256, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(
        dense_1
    )

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=["acc"])
    model.summary()
    return model


def get_arythmia_baseline_model() -> models.Model:
    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(
        16, kernel_size=5, activation=activations.relu, padding="valid"
    )(inp)
    img_1 = Convolution1D(
        16, kernel_size=5, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        32, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(
        256, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = Convolution1D(
        256, kernel_size=3, activation=activations.relu, padding="valid"
    )(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(
        dense_1
    )

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(
        optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=["acc"]
    )
    model.summary()
    return model


def train_mitbih_baseline(X: np.ndarray, Y: np.ndarray) -> None:
    model = get_arythmia_baseline_model()
    file_path = "baseline_cnn_mitbih.h5"
    checkpoint = ModelCheckpoint(
        file_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early

    model.fit(
        X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1
    )


def test_mitbih_baseline(X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    model = get_arythmia_baseline_model()
    file_path = "baseline_cnn_mitbih.h5"
    model.load_weights(file_path)

    pred_test_scores = model.predict(X_test)
    pred_test = np.argmax(pred_test_scores, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")

    print("Test f1 score : %s " % f1)

    acc = accuracy_score(Y_test, pred_test)

    print("Test accuracy score : %s " % acc)
    return pred_test_scores


def train_PTBDB_baseline(X: np.ndarray, Y: np.ndarray) -> None:
    model = get_PTBDB_baseline_model()
    file_path = "baseline_cnn_ptbdb.h5"
    checkpoint = ModelCheckpoint(
        file_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early

    model.fit(
        X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1
    )


def test_PTBDB_baseline(X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    model = get_PTBDB_baseline_model()
    file_path = "baseline_cnn_ptbdb.h5"
    model.load_weights(file_path)

    pred_test_scores = model.predict(X_test)
    pred_test = (pred_test_scores > 0.5).astype(np.int8)

    f1 = f1_score(Y_test, pred_test)

    print("Test f1 score : %s " % f1)

    acc = accuracy_score(Y_test, pred_test)

    print("Test accuracy score : %s " % acc)

    return pred_test_scores

