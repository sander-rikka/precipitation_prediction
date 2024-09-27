import pandas as pd
from src.models.avely_lstm import simple_lstm_v1, train_model, test_model


def load_data():
    df = pd.read_csv('combined_data.csv')
    df = df.dropna()
    train_data = df.sample(frac=0.75, random_state=137)
    test_data = df.drop(train_data.index)

    train_feature = train_data.copy()
    test_feature = test_data.copy()

    train_labels = train_feature.pop('WindSpeed')
    test_labels = test_feature.pop('WindSpeed')

    train_feature = train_feature.drop(
        columns=['Datetime', 'ERA5_Datetime', 'WindDirection', 'Temperature', 'MeanSeaLevelPressure',
                 'MeanWaveDirection', 'MeanWavePeriod', 'SignificantHeight', 'Latitude', 'Longitude'])
    test_feature = test_feature.drop(
        columns=['Datetime', 'ERA5_Datetime', 'WindDirection', 'Temperature', 'MeanSeaLevelPressure',
                 'MeanWaveDirection', 'MeanWavePeriod', 'SignificantHeight', 'Latitude', 'Longitude'])

    return train_feature, test_feature, train_labels, test_labels


def main():
    x_train, x_test, y_train, y_test = load_data()
    print('Got the data, trying to train.')

    data_train = x_train.values
    data_train_reshape = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])

    data_test = x_test.values
    data_test_reshape = data_test.reshape(data_test.shape[0], 1, data_test.shape[1])

    model = simple_lstm_v1((1, data_train_reshape.shape[2]))
    trained_model = train_model(model, data_test_reshape, y_test)

    rho = test_model(trained_model, data_test_reshape, y_test)
    print(f'Correlation of ERA5 wind speed with estimated wind speed from wave spectra is {rho}.')
    return None


if __name__ == '__main__':
    main()
