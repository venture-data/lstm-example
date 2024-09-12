import sys
import pandas as pd
from prophet import Prophet
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_model(model_path):
    """ Load the trained Prophet model from the specified path """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded Prophet model from '{model_path}'.")
        return model
    except FileNotFoundError:
        print(f"Error: Trained model file '{model_path}' not found.")
        sys.exit(1)

def update_features(data, current_prediction):
    """ Update lag, EMA, and rolling window features with the current prediction """
    # Append the predicted 'yhat' to the dataset
    data = pd.concat([data, current_prediction])

    # Update lag features
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        data[f'lag_{lag}'] = data['y'].shift(lag)

    # Update rolling window features
    windows = [3, 6, 12, 24, 168]
    for window in windows:
        data[f'rolling_mean_{window}h'] = data['y'].rolling(window=window).mean()
        data[f'rolling_std_{window}h'] = data['y'].rolling(window=window).std()

    # Update EMA features
    ema_windows = [12, 24, 168]
    for window in ema_windows:
        data[f'ema_{window}h'] = data['y'].ewm(span=window).mean()

    # Handle NaN values appropriately (skip if not enough data)
    data = data.ffill().bfill()  # Forward and backward fill to handle NaNs
    
    return data

def preprocess_data(data, start_date, end_date):
    """ Preprocess the input data for forecasting """
    # Convert 'Date' to datetime and handle errors
    if 'Date' in data.columns:
        data['ds'] = pd.to_datetime(data['Date'], errors='coerce').dt.round('h')
        data = data.drop(columns=['Date'])
    else:
        print("Error: 'Date' column not found in input data.")
        sys.exit(1)

    # print(f"Data types after conversion:\n{data.dtypes}")  # Check data types

    # Check if 'PriceHU' exists and set it as the target variable
    if 'PriceHU' in data.columns:
        print("'PriceHU' column found in data. Setting it as the target variable 'y'.")
        data['y'] = data['PriceHU']
        data = data.drop(columns=['PriceHU'])
    else:
        print("Warning: 'PriceHU' column not found. Proceeding without target variable 'y'.")

    # Check for NaT values in the 'ds' column
    num_nat = data['ds'].isna().sum()
    if num_nat > 0:
        print(f"Number of NaT values in 'ds' column after conversion: {num_nat}. Removing these rows...")
        data = data.dropna(subset=['ds'])
        print(f"Removed rows with NaT values. Remaining rows: {len(data)}.")

    # Check the range of dates
    print(f"Min ds: {data['ds'].min()}, Max ds: {data['ds'].max()}")  # Check the range of dates

    # Filter data to only include the required prediction dates
    future = data[(data['ds'] >= start_date) & (data['ds'] <= end_date)].copy()
    
    if future.empty:
        print("Error: The future DataFrame is empty after filtering. Check the date range.")
        print(f"Date range provided: {start_date} to {end_date}")
        print(f"Available data date range: {data['ds'].min()} to {data['ds'].max()}")
        sys.exit(1)

    # Define regressor columns used during training
    regressor_columns = [col for col in data.columns if col not in ['ds', 'y']]

    return future, regressor_columns

def iterative_forecast(model, data, start_date, end_date, regressor_columns):
    """ Perform iterative forecasting with dynamic feature updates """
    # Prepare the data for iterative forecasting
    forecast_df = pd.DataFrame(columns=['ds', 'yhat'])
    current_data = data.copy()

    # Predict iteratively for each time step
    for current_date in pd.date_range(start=start_date, end=end_date, freq='h'):
        future_df = current_data[(current_data['ds'] == current_date)].copy()

        # Ensure no NaNs in regressor columns before prediction
        if future_df[regressor_columns].isna().sum().sum() > 0:
            future_df[regressor_columns] = future_df[regressor_columns].fillna(0)

        # Predict the next time step
        forecast = model.predict(future_df)

        # Append the prediction to forecast_df
        forecast_df = pd.concat([forecast_df, forecast[['ds', 'yhat']]], ignore_index=True)

        # Update the 'y' column with the predicted value for the next iteration
        current_data.at[len(current_data), 'y'] = forecast['yhat'].iloc[0]
        
        # Update lag, rolling window, and EMA features
        current_data = update_features(current_data, forecast)

    return forecast_df

def main(csv_file, start_date_str, end_date_str):
    # Convert command-line dates to datetime
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Ensure start_date is before end_date
    if start_date > end_date:
        print("Error: Start date must be before end date.")
        sys.exit(1)

    # Load the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'prophet_model.pkl')
    model = load_model(model_file)

    # Load data from CSV file
    data = pd.read_csv(csv_file)
    print(f"Data loaded from {csv_file}. Number of rows: {len(data)}, Number of columns: {len(data.columns)}")

    # Preprocess the data
    future, regressor_columns = preprocess_data(data, start_date, end_date)

    # Forecast iteratively
    forecast_df = iterative_forecast(model, future, start_date, end_date, regressor_columns)

    # Save forecast to CSV
    forecast_output_file = os.path.join(script_dir, 'prophet_forecast.csv')
    forecast_df.to_csv(forecast_output_file, index=False)
    print(f"Forecast results saved to {forecast_output_file}.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <csv_file> <start_date> <end_date>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])