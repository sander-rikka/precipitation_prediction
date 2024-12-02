from src.data.helena_data import get_train_test_data
from src.models.helena_lstm import simple_lstm_v1, train_model
from sklearn.model_selection import train_test_split


def splitting_wrapper(input_data, target_data):
    x_train, x_rem, y_train, y_rem = train_test_split(input_data, target_data, train_size=0.7, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def main_start():
    train_x, train_y = get_train_test_data()
    train_y = train_y.reshape([train_y.shape[0], train_y.shape[-1]])
    print('Got the data!')
    X_train, X_test, X_valid, y_train, y_test, y_valid = splitting_wrapper(train_x, train_y)

    xrows, xcols, xdim = X_train.shape
    input_size = (xcols, xdim)
    model = simple_lstm_v1(input_size)
    trained_model = train_model(model, X_train, y_train, X_test, y_test, mdl_name='first_helena_test')
    print('first training complete! now write some testing functions. ')


if __name__ == '__main__':
    main_start()