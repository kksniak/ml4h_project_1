import tensorflow as tf


def get_lstm_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(187, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['acc'])

    return model
