import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def simple_lstm_v1(input_size):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=input_size, return_sequences=False))
    model.add(Dense(3 * 2, activation='softmax'))  # Assuming binary classification (precipitation or no precipitation)
    model.add(Reshape((3, 2)))  # Reshape to (3, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    return model


def train_model(model, X_train, y_train, mdl_name=None):

    if not mdl_name:
        systime = datetime.now()
        systimestr = datetime.strftime(systime, '%Y%m%dT%H%M')
        mdl_name = f'{systimestr}_simple_lstm'

    callbacks = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto',
                              baseline=None, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train,
                        y_train,
                        callbacks=[callbacks],
                        epochs=50,
                        batch_size=32,
                        validation_split=0.3)

    results_df = pd.DataFrame(history.history)
    results_csv_file = f'{mdl_name}_history.csv'
    with open(results_csv_file, mode='w') as f:
        results_df.to_csv(f)

    model.save(f'./{mdl_name}_model.h5')

    return model


def test_model(model, X_test, y_test):

    y_hat = model.predict(X_test, verbose=0)
    accuracy = categorical_crossentropy(y_test, y_hat)
    """
    Write what ever goodness descriptor possible and/or necessary.
    """
    return accuracy