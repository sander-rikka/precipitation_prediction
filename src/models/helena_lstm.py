import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras


def root_mean_squared_error(y_true, y_pred):
    squared_difference = K.mean(tf.square(K.exp(y_true) - K.exp(y_pred)), axis=1)
    return tf.sqrt(squared_difference)


def rho(y_true, y_pred):
    return tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)


def simple_lstm_v1(input_size):
    # Define the LSTM model
    print(input_size)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(64, input_shape=input_size, return_sequences=True))
    model.add(LSTM(64, activation='tanh'))
    model.add(Dense(31))

    #model.compile(loss=root_mean_squared_error, optimizer=opt, metrics=[rho])
    model.compile(loss='mse')
    model.summary()

    return model


def train_model(model, X_train, y_train, X_test, y_test, mdl_name=None):

    systime = datetime.now()
    systimestr = datetime.strftime(systime, '%Y%m%dT%H%M')
    if not mdl_name:
        mdl_name = f'{systimestr}_simple_lstm'
    else:
        mdl_name = f'{systimestr}_{mdl_name}'

    callbacks = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_test, y_test),
                        epochs=25,
                        batch_size=32)

    results_df = pd.DataFrame(history.history)
    results_csv_file = f'{mdl_name}_history.csv'
    with open(results_csv_file, mode='w') as f:
        results_df.to_csv(f)

    model.save(f'./{mdl_name}_model.h5')

    return model


def test_model(model, X_test, y_test):

    y_hat = model.predict(X_test, verbose=0)
    accuracy = np.corrcoef(y_test, y_hat[:,0])[0, 1]
    """
    Write what ever goodness descriptor possible and/or necessary.
    """
    return accuracy
