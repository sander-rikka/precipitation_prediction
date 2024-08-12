"""Start of main program."""
from src.data.data_load import load_samples, get_train_samples
from src.models.simple_lstm import simple_lstm_v1, train_model, test_model
from src.models.tune_columns_optuna import tune_columns
from src.models.tune_model_optuna import tune_model


def train_simple_lstm():
    data_path = r'C:\Temp'
    fname = 'samples.npz'
    X_train, X_test, y_train, y_test = get_train_samples(load_samples(data_path, fname))

    samples, n_timestep, n_features = X_train.shape
    model = simple_lstm_v1((n_timestep, n_features))
    train_model(model, X_train, y_train)
    test_model(model, X_test, y_test)
    return None


if __name__ == '__main__':
    #train_simple_lstm()
    tune_columns()
    #tune_model()

    pass

