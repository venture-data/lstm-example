import pandas as pd
import numpy as np
import sys
from prophet import Prophet
from sklearn.preprocessing import RobustScaler
import joblib
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def add_moving_averages(df, column_list, windows):
    logging.debug("Starting to calculate moving averages.")
    # Create a dictionary to store new columns for moving averages
    ma_dict = {}
    for column in column_list:
        for window in windows:
            ma_dict[f"{column}_ma_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
            logging.debug(f"Calculated moving average for {column} with window size {window}.")
    # Use pd.concat to add all columns at once
    logging.debug("Completed calculating moving averages.")
    return pd.concat([df, pd.DataFrame(ma_dict)], axis=1)

def feature_engineering(df):
    logging.debug("Starting feature engineering.")
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
    logging.debug("Feature engineering completed.")
    return df

def fill_missing_values(df, target, regressors):
    logging.debug("Starting to fill missing values.")
    # Spline interpolation for target and regressors
    df[target].interpolate(method='spline', order=3, inplace=True)
    logging.debug(f"Filled missing values for target: {target} using spline interpolation.")

    for regressor in regressors:
        df[regressor].interpolate(method='spline', order=3, inplace=True)
        logging.debug(f"Filled missing values for regressor: {regressor} using spline interpolation.")
    logging.debug("Completed filling missing values.")

def split_data(df, train_start_date, train_end_date, val_ratio=0.1, test_ratio=0.05):
    logging.debug("Starting to split data into train, validation, and test sets.")
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

    logging.debug("Data splitting completed.")
    return train_df, val_df, test_df

def main(input_file, start_date, end_date, regressors):
    logging.info("Starting the model training process.")

    # Load the data
    logging.debug("Loading data from Excel file.")
    df = pd.read_excel(input_file)
    logging.info(f"Data loaded with shape: {df.shape}")

    # Convert date column to datetime with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
    logging.debug("Converted 'Date' column to datetime.")

    # Filter data between start_date and end_date
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    logging.info(f"Data filtered between {start_date} and {end_date}. Remaining rows: {df.shape[0]}")

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
    logging.debug(f"Applying moving averages with window sizes: {windows}")
    df = add_moving_averages(df, [target] + regressor_list, windows)

    # Drop rows with NaN values
    df.dropna(inplace=True)
    logging.info(f"Data after dropping NaN values: {df.shape}")

    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data(df, start_date, end_date)

    logging.info(f"Train set size: {train_df.shape}, Validation set size: {val_df.shape}, Test set size: {test_df.shape}")

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
    logging.info("Model fitting completed.")

    # Save the trained model to a file
    model_filename = 'trained_prophet_model.pkl'
    joblib.dump(model, model_filename)
    logging.info(f"Model training complete. The model has been saved to '{model_filename}'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prophet_training.py <input_file> <start_date> <end_date> <regressors>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    start_date = pd.to_datetime(sys.argv[2], format='%Y-%m-%d %H:%M')
    end_date = pd.to_datetime(sys.argv[3], format='%Y-%m-%d %H:%M')
    regressors = sys.argv[4]

    main(input_file, start_date, end_date, regressors)
