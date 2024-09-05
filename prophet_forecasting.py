import pandas as pd
import numpy as np
import sys
from prophet import Prophet
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main(input_file, predict_start_date, predict_end_date, regressors):
    logging.info("Starting the forecasting process.")

    # Load the trained model
    model_filename = 'trained_prophet_model.pkl'
    logging.info(f"Loading trained model from '{model_filename}'.")
    model = joblib.load(model_filename)

    # Load the data for future forecasting
    logging.debug("Loading data from Excel file.")
    df = pd.read_excel(input_file)
    logging.info(f"Data loaded with shape: {df.shape}")

    # Convert date column to datetime with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
    df = df.rename(columns={'Date': 'ds'})  # Adjust date column name for consistency

    # Convert the regressors string to a list
    regressor_list = regressors.strip('[]').split(',')
    logging.debug(f"Regressors identified: {regressor_list}")

    # Create a future dataframe for the prediction period
    future_dates = pd.date_range(start=predict_start_date, end=predict_end_date, freq='H')
    future = pd.DataFrame({'ds': future_dates})
    logging.debug("Preparing future data for forecasting.")

    # Merge regressor data into the future dataframe
    future = future.merge(df[['ds'] + regressor_list], on='ds', how='left')

    # Check for any NaN values after merging
    for regressor in regressor_list:
        missing_count = future[regressor].isna().sum()
        if missing_count > 0:
            logging.error(f"After merging, regressor {regressor} has {missing_count} missing values.")
            print(f"Warning: {missing_count} missing values in {regressor} after merging. Consider filling these.")
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
    logging.info("Forecasting future values.")
    forecast = model.predict(future)
    logging.info("Forecasting complete.")

    # Output the forecast results
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_output.csv', index=False)
    logging.info("Forecast results saved to 'forecast_output.csv'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prophet_forecasting.py <input_file> <predict_start_date> <predict_end_date> <regressors>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    predict_start_date = pd.to_datetime(sys.argv[2], format='%Y-%m-%d %H:%M')
    predict_end_date = pd.to_datetime(sys.argv[3], format='%Y-%m-%d %H:%M')
    regressors = sys.argv[4]

    main(input_file, predict_start_date, predict_end_date, regressors)
