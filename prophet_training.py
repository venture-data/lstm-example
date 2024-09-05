import pandas as pd
import numpy as np
import sys
from prophet import Prophet
from sklearn.preprocessing import RobustScaler
import json

def add_moving_averages(df, column_list, windows):
    # Create a dictionary to store new columns for moving averages
    ma_dict = {}
    for column in column_list:
        for window in windows:
            ma_dict[f"{column}_ma_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
    # Use pd.concat to add all columns at once
    return pd.concat([df, pd.DataFrame(ma_dict)], axis=1)

def feature_engineering(df):
    # Extract datetime features
    df['month'] = df['ds'].dt.month
    df['week'] = df['ds'].dt.weekday
    df['is_weekend'] = df['ds'].dt.weekday.isin([5, 6]).astype(int)
    df['hour'] = df['ds'].dt.hour
    df['peak_event'] = ((df['ds'].dt.year.isin([2023, 2022])) | (df['ds'] >= pd.to_datetime("2024-06-01"))).astype(int)

    # One-hot encoding for time-based features
    for i in range(12):
        df[f"month_{i}"] = (df['ds'].dt.month == i).astype(int)
    for i in range(7):
        df[f"week_{i}"] = (df['ds'].dt.weekday == i).astype(int)
    for i in range(24):
        df[f"hour_{i}"] = (df['ds'].dt.hour == i).astype(int)
    
    return df

def fill_missing_values(df, target, regressors):
    # Spline interpolation for target and regressors
    df[target].interpolate(method='spline', order=3, inplace=True)

    for regressor in regressors:
        df[regressor].interpolate(method='spline', order=3, inplace=True)

def split_data(df, train_start_date, train_end_date, val_ratio=0.1, test_ratio=0.05):
    total_days = (train_end_date - train_start_date).days
    train_days = int(total_days * (1 - val_ratio - test_ratio))
    val_days = int(total_days * val_ratio)
    test_days = total_days - train_days - val_days

    val_start_date = train_start_date + pd.Timedelta(days=train_days)
    val_end_date = val_start_date + pd.Timedelta(days=val_days)
    test_start_date = val_end_date + pd.Timedelta(days=1)
    test_end_date = test_start_date + pd.Timedelta(days=test_days)

    train_df = df[(df['ds'] >= train_start_date) & (df['ds'] <= val_start_date)]
    val_df = df[(df['ds'] > val_start_date) & (df['ds'] <= val_end_date)]
    test_df = df[(df['ds'] > val_end_date) & (df['ds'] <= test_end_date)]

    return train_df, val_df, test_df

def main(input_file, start_date, end_date, regressors):
    # Load the data
    df = pd.read_excel(input_file)

    # Convert date column to datetime with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')

    # Filter data between start_date and end_date
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Prepare the data for Prophet
    df = df.rename(columns={'Date': 'ds', 'PriceSK': 'y'})  # Adjust target variable name
    target = 'y'

    # Convert the regressors string to a list
    regressor_list = regressors.strip('[]').split(',')

    # Fill missing values in the target and regressors using spline interpolation
    fill_missing_values(df, target, regressor_list)

    # Apply feature engineering
    df = feature_engineering(df)

    # Moving average window sizes
    windows = [12, 24, 36, 48, 7 * 24]
    print(f"Applying moving averages with window sizes: {windows}")
    df = add_moving_averages(df, [target] + regressor_list, windows)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data(df, start_date, end_date)

    print(f"Train set size: {train_df.shape}, Validation set size: {val_df.shape}, Test set size: {test_df.shape}")

    # Initialize the Prophet model
    model = Prophet()

    # Add additional regressors
    for regressor in regressor_list:
        model.add_regressor(regressor)

    # Fit the model
    model.fit(train_df)

    # Forecasting future values
    future = model.make_future_dataframe(periods=365, freq='H')

    # Add the regressors for the future data
    for regressor in regressor_list:
        future[regressor] = train_df[regressor].values[:len(future)]

    # Predict future values
    forecast = model.predict(future)

    # Output the forecast results
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_output.csv', index=False)
    print("Forecasting complete. Results saved to 'forecast_output.csv'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prophet_training.py <input_file> <start_date> <end_date> <regressors>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    start_date = pd.to_datetime(sys.argv[2], format='%Y-%m-%d %H:%M')
    end_date = pd.to_datetime(sys.argv[3], format='%Y-%m-%d %H:%M')
    regressors = sys.argv[4]

    main(input_file, start_date, end_date, regressors)
