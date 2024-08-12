import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_samples(path, fname) -> tuple[np.ndarray, np.ndarray]:
    print('Hi. This is data loading.')
    fpath = os.path.join(path, fname)

    with np.load(rf'{fpath}', allow_pickle=True) as data:
        harku_samples_x = data['harku_samples_x']
        harku_samples_y = data['harku_samples_y']
        tartu_samples_x = data['tartu_samples_x']
        tartu_samples_y = data['tartu_samples_y']

    data_x = np.vstack([harku_samples_x, tartu_samples_x])
    data_y = np.vstack([harku_samples_y, tartu_samples_y])

    return data_x, data_y


def get_train_samples(X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print('Get your samples ready.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=137)

    return X_train, X_test, y_train, y_test
