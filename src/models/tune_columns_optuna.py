import optuna
import numpy as np
from datetime import datetime
from src.data.data_load import load_samples, get_train_samples
from src.models.simple_lstm import simple_lstm_v1
from tensorflow.keras.losses import categorical_crossentropy


def objective(trial):
    n_features = 30
    columns = np.arange(n_features, dtype=int)

    # Select a subset of features
    selected_features = [col for col in columns if trial.suggest_categorical(col, [True, False])]
    if len(selected_features) == 0:
        return float('inf')  # Return a large value if no features are selected

    batch_size = trial.suggest_int('batch_size_exponent', 2, 8)
    batch_size = 2 ** batch_size

    # add precipitation column. maybe wind u,v should also be added, maybe something else critical??
    if 0 not in selected_features:
        selected_features.append(0)

    data_path = r'C:\Temp'
    fname = 'samples.npz'
    data_x, data_y = load_samples(data_path, fname)

    X = data_x[:, :, selected_features]
    X_train, X_test, y_train, y_test = get_train_samples(X, data_y)

    samples, n_timestep, n_features = X_train.shape
    model = simple_lstm_v1((n_timestep, n_features))

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=2, validation_split=0.2)

    # Evaluate the model
    preds = model.predict(X_test)
    cat_loss = categorical_crossentropy(y_test, preds)

    return cat_loss


def tune_columns(mdl_name=None):
    print('Hi, this is to find the most optimal features for simple LSTM.')
    if not mdl_name:
        systime = datetime.now()
        systimestr = datetime.strftime(systime, '%Y%m%dT%H%M')
        mdl_name = f'{systimestr}_simple_lstm'

    # Create the study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters and mean squared error
    print("Best hyperparameters: ", study.best_params)
    print("Best mean squared error: ", study.best_value)

    # Retrieve the best features
    n_features = 30
    columns = np.arange(n_features, dtype=int)
    best_features = [col for col in columns if study.best_params[col]]
    print("Best features: ", best_features)

    # Prepare the data with the best features
    data_path = r'C:\Temp'
    fname = 'samples.npz'
    data_x, data_y = load_samples(data_path, fname)

    X = data_x[:, :, best_features]
    X_train, X_test, y_train, y_test = get_train_samples(X, data_y)

    # Save model
    samples, n_timestep, n_features = X_train.shape
    best_model = simple_lstm_v1((n_timestep, n_features))

    batch_size = 2 ** study.best_params['batch_size_exponent']
    best_model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=2, validation_split=0.2)
    best_model.save(f"{mdl_name}_best_lstm_model_optimized_features.h5")

    return None

