import pandas as pd
import numpy as np


# Updated function to use '2m_Temperature', 'MeanSeaLevelPressure', etc.
def time_difference_check(df, index, past_range, future_range):
    data_range = past_range + future_range
    timestamps = df['Datetime'].iloc[index: index + data_range]  # Use 'Datetime'
    time_diffs = timestamps.diff()
    check = (time_diffs[1:] == pd.Timedelta(hours=1)).all()
    return check


def one_sample(df, index, past_range, future_range):
    # Select past data (features for the model, drop 'Datetime')
    future_data = df.iloc[index + past_range: index + past_range + future_range].copy()  # Use .copy() to avoid SettingWithCopyWarning
    future_data.drop(columns=['Datetime'], inplace=True)
    future_data = future_data.iloc[:, 11:72:2]

    # Select future data (target variables: temperature, pressure, wind components)
    past_data = df.iloc[index: index + past_range + future_range][
        ['2m_Temperature', 'MeanSeaLevelPressure', 'Ucomponentofwind', 'Vcomponentofwind']]

    # Convert future data to float if necessary
    future_data = future_data.astype(float)

    # Ensure columns are in the correct order
    past_data = past_data[['2m_Temperature', 'MeanSeaLevelPressure', 'Ucomponentofwind', 'Vcomponentofwind']]

    return past_data.values, future_data.values


def training_samples(df, past_range, future_range):
    train_x = []
    train_y = []

    for i in range(len(df) - past_range - future_range):
        if time_difference_check(df, i, past_range, future_range):
            sample_data_x, sample_data_y = one_sample(df, i, past_range, future_range)
            train_x.append(sample_data_x)
            train_y.append(sample_data_y)

    return train_x, train_y


def get_train_test_data():
    # Load the data (make sure to replace this with your actual file path)
    all_data = pd.read_csv('src/data/merged_selka_era5_data_spec_param.csv')  # Replace with your actual data file path
    all_data = all_data.dropna()

    # Ensure 'Datetime' column is in datetime format
    all_data['Datetime'] = pd.to_datetime(all_data['Datetime'])  # Ensure it's in datetime format

    # Set the past and future r anges
    past_hours = 11  # Example: Use past 3 hours of data
    future_hours = 1  # Example: Predict for the next 1 hour

    # Generate the samples
    samples_x, samples_y = training_samples(all_data, past_hours, future_hours)

    # Check the shape and content of the first sample
    print("First sample of past data (X):", samples_x[0])
    print("First sample of future data (Y):", samples_y[0])

    # Optionally check the length of your data
    print(f"Total samples: {len(samples_x)}")
    return np.array(samples_x), np.array(samples_y)


if __name__ == '__main__':
    get_train_test_data()