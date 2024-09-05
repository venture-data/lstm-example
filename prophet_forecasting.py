import pandas as pd
import numpy as np
import sys
from prophet import Prophet
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_future_data(df, start_date, end_date, regressors):
    """
    Prepare the future DataFrame for forecasting with the given date range and regressors.
    """
    logging.debug("Preparing future data for forecasting.")

    # Generate a date range for future predictions
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    future = pd.DataFrame({'ds': future_dates})

    # Merge the future DataFrame with the input data to ensure regressors are aligned
    future = future.merge(df[['ds'] + regressors], on='ds', how='left')

    # Fill missing regressor values with the last known value or interpolate if needed
    for regressor in regressors:
        future[regressor].interpolate(method='linear', inplace=True)
        future[regressor].fillna(method='ffill', inplace=True)
        logging.debug(f"Filled missing values for regressor: {regressor}")

    logging.debug("Future data prepared for forecasting.")
    return future

def main(input_file, start_date, end_date, regressors):
    logging.info("Starting the forecasting process.")

    # Load the trained model
    model_filename = 'trained_prophet_model.pkl'
    model = joblib.load(model_filename)
    logging.info(f"Loaded trained model from '{model_filename}'.")

    # Load the data
    logging.debug("Loading data from Excel file.")
    df = pd.read_excel(input_file)
    logging.info(f"Data loaded with shape: {df.shape}")

    # Convert date column to datetime with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
    df = df.rename(columns={'Date': 'ds'})  # Rename for consistency with Prophet

    # Convert the regressors string to a list
    regressor_list = regressors.strip('[]').split(',')
    logging.debug(f"Regressors identified: {regressor_list}")

    # Prepare the future DataFrame with the date range and regressors
    future = prepare_future_data(df, pd.to_datetime(start_date), pd.to_datetime(end_date), regressor_list)

    # Forecast future values using the loaded model
    logging.info("Forecasting future values.")
    forecast = model.predict(future)

    # Save the forecast results to a file
    forecast_output_file = 'forecast_output.csv'
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_output_file, index=False)
    logging.info(f"Forecasting complete. Results saved to '{forecast_output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prophet_forecasting.py <input_file> <start_date> <end_date> <regressors>")
        sys.exit(1)

    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    regressors = sys.argv[4]

    main(input_file, start_date, end_date, regressors)
