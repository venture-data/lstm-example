import sys
import os
import pickle
import numpy as np
import pandas as pd
from prophet import Prophet
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression

# Set file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'prophet_model.pkl')
feature_csv_file = os.path.join(script_dir, 'features_for_Prophet.csv')

# Feature Engineering Functions
def add_lagged_features(data, column, lags):
    """ Add lagged features to the data """
    for lag in lags:
        data[f'lag_{lag}'] = data[column].shift(lag)
    return data

def add_rolling_window_features(data, column, windows):
    """ Add rolling window statistics to the data """
    for window in windows:
        data[f'rolling_mean_{window}h'] = data[column].rolling(window).mean()
        data[f'rolling_std_{window}h'] = data[column].rolling(window).std()
    return data

def add_exponential_moving_average(data, column, ema_windows):
    """ Add exponential moving averages to the data """
    for window in ema_windows:
        data[f'ema_{window}h'] = data[column].ewm(span=window).mean()
    return data

def create_fourier_features(data, period, order):
    """ Create Fourier series features for seasonality """
    t = np.arange(len(data))
    for i in range(1, order + 1):
        data[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        data[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)

def iterative_forecast(model, data, start_date, end_date, regressor_columns):
    """ Perform iterative forecasting with dynamic feature updates """
    forecast_df = pd.DataFrame(columns=['ds', 'yhat'])
    current_data = data.copy()

    for current_date in pd.date_range(start=start_date, end=end_date, freq='h'):
        # Prepare future dataframe for the next step
        future_df = current_data[current_data['ds'] == current_date].copy()

        if future_df.empty:
            print(f"No data for current date: {current_date}. Skipping this step.")
            continue

        # Ensure not to use future 'PriceHU' values, instead use the predicted values
        future_df[regressor_columns] = future_df[regressor_columns].ffill().fillna(0)

        # Make prediction
        forecast = model.predict(future_df)
        forecast_df = pd.concat([forecast_df, forecast[['ds', 'yhat']]], ignore_index=True)

        # Update 'y' with predicted value for the next iteration
        predicted_value = forecast['yhat'].iloc[0]
        new_row = {'ds': current_date, 'y': predicted_value}
        current_data = pd.concat([current_data, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        # Update lag, rolling window, and EMA features using only past and predicted 'y' values
        current_data = add_lagged_features(current_data, 'y', [1, 3, 6, 12, 24, 48, 72, 168])
        current_data = add_rolling_window_features(current_data, 'y', [3, 6, 12, 24, 168])
        current_data = add_exponential_moving_average(current_data, 'y', [12, 24, 168])

    return forecast_df

def main(input_file, start_date, end_date):
    # Load and preprocess data
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['ds'] = df['Date'].dt.round('h')
    df['y'] = df['PriceHU']
    df.drop(columns=['Date', 'PriceHU'], inplace=True)

    # Verify date range
    min_date = df['ds'].min()
    max_date = df['ds'].max()

    if start_date < min_date or end_date > max_date:
        raise ValueError(f"Date range [{start_date}, {end_date}] is out of bounds.")

    # Load trained model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Define regressor columns
    regressor_columns = [col for col in df.columns if col not in ['ds', 'y']]

    # Perform iterative forecasting
    forecast_df = iterative_forecast(model, df, start_date, end_date, regressor_columns)

    # Save forecast results
    forecast_output_file = os.path.join(script_dir, 'prophet_forecast.csv')
    forecast_df.to_csv(forecast_output_file, index=False)
    print(f"Forecast results saved to {forecast_output_file}.")

if __name__ == "__main__":
    input_file = sys.argv[1]
    start_date = pd.to_datetime(sys.argv[2])
    end_date = pd.to_datetime(sys.argv[3])
    main(input_file, start_date, end_date)