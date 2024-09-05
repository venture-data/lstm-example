import pandas as pd
import numpy as np
import sys
from prophet import Prophet
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def add_moving_averages(df, column, windows):
    logging.debug("Calculating moving averages.")
    ma_dict = {}
    for window in windows:
        ma_dict[f"{column}_ma_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
        logging.debug(f"Moving average calculated for column: {column} with window size: {window}.")
    logging.debug("All moving averages calculated.")
    return pd.concat([df, pd.DataFrame(ma_dict)], axis=1)

def feature_engineering(df):
    logging.debug("Starting feature engineering.")
    df['month'] = df['ds'].dt.month
    df['week'] = df['ds'].dt.weekday
    df['is_weekend'] = df['ds'].dt.weekday.isin([5, 6]).astype(int)
    df['hour'] = df['ds'].dt.hour
    df['peak_event'] = ((df['ds'].dt.year.isin([2023, 2022])) | (df['ds'] >= pd.to_datetime("2024-06-01"))).astype(int)

    for i in range(12):
        df[f"month_{i}"] = (df['ds'].dt.month == i).astype(int)
    for i in range(7):
        df[f"week_{i}"] = (df['ds'].dt.weekday == i).astype(int)
    for i in range(24):
        df[f"hour_{i}"] = (df['ds'].dt.hour == i).astype(int)
    
    logging.debug("Feature engineering completed.")
    return df

def fill_missing_values(df, target, regressors):
    logging.debug("Filling missing values using spline interpolation.")
    df[target] = df[target].interpolate(method='spline', order=3)
    logging.debug(f"Filled missing values for target: {target}.")

    for regressor in regressors:
        df[regressor] = df[regressor].interpolate(method='spline', order=3)
        logging.debug(f"Filled missing values for regressor: {regressor}.")
    logging.debug("Missing value filling completed.")

def split_data(df, train_start_date, train_end_date):
    logging.debug("Splitting data into training set.")
    train_df = df[(df['ds'] >= train_start_date) & (df['ds'] <= train_end_date)]
    logging.debug(f"Training data range: {train_start_date} to {train_end_date}. Size: {train_df.shape}")
    return train_df

def main(input_file, train_start_date, train_end_date, regressors):
    logging.info("Starting the Prophet model training process.")

    # Load the data
    logging.debug("Loading data from Excel file.")
    df = pd.read_excel(input_file)
    logging.info(f"Data loaded with shape: {df.shape}")

    # Convert date column to datetime with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
    logging.debug("Converted 'Date' column to datetime format.")

    # Filter data between start_date and end_date
    df = df[(df['Date'] >= train_start_date) & (df['Date'] <= train_end_date)]
    logging.info(f"Data filtered between {train_start_date} and {train_end_date}. Remaining rows: {df.shape[0]}")

    # Prepare the data for Prophet
    df = df.rename(columns={'Date': 'ds', 'PriceSK': 'y'})  # Adjust target variable name
    target = 'y'

    # Convert the regressors string to a list
    regressor_list = regressors.strip('[]').split(',')
    logging.debug(f"Regressors identified: {regressor_list}")

    # Fill missing values in the target and regressors using spline interpolation
    fill_missing_values(df, target, regressor_list)

    # Apply feature engineering
    df = feature_engineering(df)

    # Moving average window sizes
    windows = [12, 24, 36, 48, 7 * 24]
    logging.debug(f"Applying moving averages to the target with window sizes: {windows}")
    df = add_moving_averages(df, target, windows)

    # Drop rows with NaN values
    df.dropna(inplace=True)
    logging.info(f"Data after dropping NaN values: {df.shape}")

    # Split data into training set
    train_df = split_data(df, train_start_date, train_end_date)

    # Initialize the Prophet model
    model = Prophet()
    logging.debug("Initialized Prophet model.")

    # Add additional regressors
    for regressor in regressor_list:
        model.add_regressor(regressor)
        logging.debug(f"Added regressor to model: {regressor}")

    # Fit the model
    logging.info("Fitting the Prophet model.")
    model.fit(train_df)

    # Forecasting future values
    future = model.make_future_dataframe(periods=(pd.to_datetime('2024-08-20') - pd.to_datetime('2024-07-01')).days * 24, freq='h')
    logging.debug("Created future DataFrame for forecasting.")

    # Add the regressors for the future data
    future = future.merge(train_df[['ds'] + regressor_list], on='ds', how='left')
    logging.debug("Merged future data with regressors.")

    # Check for any NaN values after merging
    for regressor in regressor_list:
        missing_count = future[regressor].isna().sum()
        if missing_count > 0:
            logging.error(f"After merging, regressor {regressor} has {missing_count} missing values.")
        else:
            logging.debug(f"Regressor {regressor} has no missing values after merging.")

    # Ensure no missing values before forecasting
    future.fillna(method='ffill', inplace=True)
    future.fillna(method='bfill', inplace=True)

    # Check again for NaN values after filling
    if future.isna().any().any():
        logging.error("NaN values found in future data even after filling. Exiting.")
        sys.exit(1)

    # Predict future values
    forecast = model.predict(future)
    logging.info("Forecasting complete.")

    # Output the forecast results
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_output.csv', index=False)
    logging.info("Forecast results saved to 'forecast_output.csv'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prophet_training.py <input_file> <train_start_date> <train_end_date> <regressors>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    train_start_date = pd.to_datetime(sys.argv[2], format='%Y-%m-%d %H:%M')
    train_end_date = pd.to_datetime(sys.argv[3], format='%Y-%m-%d %H:%M')
    regressors = sys.argv[4]

    main(input_file, train_start_date, train_end_date, regressors)
