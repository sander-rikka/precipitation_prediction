import optuna
from datetime import datetime
from src.data.data_load import load_samples, get_train_samples
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam


def objective(trial):
    # Load data
    data_path = r'C:\Temp'
    fname = 'samples.npz'
    data_x, data_y = load_samples(data_path, fname)
    X_train, X_test, y_train, y_test = get_train_samples(data_x, data_y)

    n_samples, n_timesteps, n_features = X_train.shape

    # Define hyperparameters to tune
    batch_size = trial.suggest_int('batch_size_exponent', 2, 8)
    batch_size = 2 ** batch_size

    n_layers = trial.suggest_int('n_layers', 1, 3)
    units0 = trial.suggest_int('units0', 16, 128, log=True)
    units_loop = trial.suggest_int('units_loop', 16, 128, log=True)
    #units1 = trial.suggest_int('units1', 16, 128, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)

    # Build the model
    model = Sequential()
    model.add(LSTM(units0, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    for i in range(n_layers - 1):
        model.add(LSTM(units_loop, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
    #model.add(LSTM(units1, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(3 * 2, activation='softmax'))  # Assuming binary classification (precipitation or no precipitation)
    model.add(Reshape((3, 2)))  # Reshape to (3, num_classes)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=2, validation_split=0.2)

    # Evaluate the model
    preds = model.predict(X_test)
    cat_loss = categorical_crossentropy(y_test, preds)

    return cat_loss


def tune_model(mdl_name=None):
    print('Hi, this is to find most optimized LSTM configuration for all features. ')
    if not mdl_name:
        systime = datetime.now()
        systimestr = datetime.strftime(systime, '%Y%m%dT%H%M')
        mdl_name = f'{systimestr}_simple_lstm'

    # Create the study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters and mean squared error
    print("Best hyperparameters: ", study.best_params)
    print("Best categorical loss error: ", study.best_value)

    # Load data
    data_path = r'C:\Temp'
    fname = 'samples.npz'
    data_x, data_y = load_samples(data_path, fname)
    X_train, X_test, y_train, y_test = get_train_samples(data_x, data_y)

    n_samples, n_timesteps, n_features = X_train.shape

    # Save the best model
    best_model = Sequential()
    best_model.add(LSTM(study.best_params['units0'], activation='relu', input_shape=(n_timesteps, n_features),
                        return_sequences=True))
    for i in range(study.best_params['n_layers'] - 1):
        best_model.add(LSTM(study.best_params['units_loop'], activation='relu', return_sequences=True))
        best_model.add(Dropout(study.best_params['dropout_rate']))
    #best_model.add(LSTM(study.best_params['units'], activation='relu'))
    #best_model.add(Dropout(study.best_params['dropout_rate']))
    best_model.add(Dense(1))
    best_model.add(Dense(3 * 2, activation='softmax'))  # Assuming binary classification (precipitation or no precipitation)
    best_model.add(Reshape((3, 2)))  # Reshape to (3, num_classes)

    # Compile the model
    best_model.compile(optimizer=Adam(learning_rate=study.best_params['learning_rate']),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model
    batch_size = 2 ** study.best_params['batch_size_exponent']
    best_model.fit(X_train, y_train, epochs=150, batch_size=batch_size, verbose=2, validation_split=0.2)

    best_model.save(f"{mdl_name}_optimized_lstm_model_allfeatures.h5")
