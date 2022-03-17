import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, Conv1D, Dropout, GlobalMaxPooling1D, Dense
import numpy as np

class Attention:


    @staticmethod
    def model(input_shape, n_classes, n_encoders = 2, key_dim=8, num_heads=8, feedforward_layers = [64]):

        inputs = keras.Input(shape=input_shape)
        x = inputs

        for _ in range(n_encoders):

            res_1 = x
            x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=0.1)(x, x)
            
            res_2 = x + res_1
            x = res_2
            for filter in feedforward_layers:
                x = Conv1D(filters=filter, kernel_size=1, activation="relu")(x)
                x = Dropout(0.1)(x)
                x = Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
                x = x + res_2

        x = GlobalMaxPooling1D(data_format="channels_first")(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(n_classes, activation="softmax")(x)

        return keras.Model(inputs, outputs)
    

    @staticmethod
    def train_arythmia_attention(x_train, y_train, x_test, y_test):
        
        tf.random.set_seed(0)
        n_classes = len(np.unique(y_train))
        model = Attention.model(x_train.shape[1:], n_classes,  n_encoders = 2, key_dim=8, num_heads=8, feedforward_layers = [64])

        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=["acc"])

        early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        redonplat = keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, mode="max", patience=3, verbose=2)
        callbacks = [early_stopping, redonplat]

        model.fit(x_train,y_train,validation_split=0.1,epochs=50,batch_size=64,callbacks=callbacks)

        return model

    @staticmethod
    def train_mitbih_attention(x_train, y_train, x_test, y_test):
        
        tf.random.set_seed(0)
        n_classes = len(np.unique(y_train))
        model = Attention.model(x_train.shape[1:], n_classes,  n_encoders = 2, key_dim=8, num_heads=8, feedforward_layers = [64])

        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=["acc"])

        early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        redonplat = keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, mode="max", patience=3, verbose=2)
        callbacks = [early_stopping, redonplat]

        model.fit(x_train,y_train,validation_split=0.1,epochs=50,batch_size=64,callbacks=callbacks)

        model.evaluate(x_test, y_test, verbose=1)

        return model

#example of usage
from datasets import load_PTB_dataset, load_arrhythmia_dataset

x, y, x_test, y_test = load_arrhythmia_dataset()
model = Attention.train_arythmia_attention(x, y, x_test, y_test)
model.evaluate(x_test, y_test, verbose=1)