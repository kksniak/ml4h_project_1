import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from datasets import load_arrhythmia_dataset
from utils import upsample, get_debug_data, evaluate_model
from tf_models import get_lstm_model

tf.random.set_seed(1337)
np.random.seed(42)

DEBUG = False
UPSAMPLE = True
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 1
EARLY_STOPPING_PATIENCE = 1

X, y, X_test, y_test = load_arrhythmia_dataset()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, stratify=y)

if DEBUG:
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_debug_data(
        X_train, y_train, X_valid, y_valid, X_test, y_test)

if UPSAMPLE:
    X_train, y_train = upsample(X_train, y_train)

models = [get_lstm_model()]

for model in models:
    model.summary()
    model.fit(X_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(X_valid, y_valid),
              shuffle=True,
              epochs=EPOCHS,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)
              ])
    evaluate_model(model, X_test, y_test)
