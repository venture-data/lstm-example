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

def preprocess_data(data, start_date, end_date):
    """ Preprocess the input data for forecasting """
    # Convert 'Date' to datetime and rename columns
    data['ds'] = pd.to_datetime(data['Date'], errors='coerce').dt.round('h')

    # Check if 'PriceHU' exists and set it as the target variable
    if 'PriceHU' in data.columns:
        print("'PriceHU' column found in data. Setting it as the target variable 'y'.")
        data['y'] = data['PriceHU']
        data = data.drop(columns=['Date', 'PriceHU'])
    else:
        data = data.drop(columns=['Date'])
    
    # Check for NaT values in the 'ds' column
    num_nat = data['ds'].isna().sum()
    if num_nat > 0:
        print(f"Number of NaT values in 'ds' column after conversion: {num_nat}. Removing these rows...")
        data = data.dropna(subset=['ds'])
        print(f"Removed rows with NaT values. Remaining rows: {len(data)}.")

    # Filter data to only include the required prediction dates
    future = data[(data['ds'] >= start_date) & (data['ds'] <= end_date)].copy()
    
    if future.empty:
        print("Error: The future DataFrame is empty after filtering. Check the date range.")
        sys.exit(1)

    # Define regressor columns used during training
    regressor_columns = [col for col in data.columns if col not in ['ds', 'y']]

    # Fill missing regressor values
    future[regressor_columns] = future[regressor_columns].ffill().bfill()

    print(f"Prepared future DataFrame for forecasting. First few rows:\n{future.head()}")

    # Check for NaN values after filling
    if future[regressor_columns].isna().sum().sum() > 0:
        print("Warning: There are still NaN values in the regressors after filling.")
        future[regressor_columns] = future[regressor_columns].fillna(0)  # Final fallback to zero

    return future, regressor_columns

def forecast_and_save(model, future, output_path):
    """ Perform forecasting and save the results to a CSV file """
    print("Starting forecast...")
    forecast = model.predict(future)
    print("Forecasting completed.")

    forecast.to_csv(output_path, index=False)
    print(f"Forecast results saved to {output_path}.")
    return forecast

def evaluate_forecast(data, forecast, start_date, end_date):
    """ Evaluate the forecast using actual data and save metrics """
    actual_data = data[(data['ds'] >= start_date) & (data['ds'] <= end_date)][['ds', 'y']].copy()
    forecast_results = forecast[['ds', 'yhat']].copy()
    evaluation_df = pd.merge(actual_data, forecast_results, on='ds', how='inner')

    # Calculate evaluation metrics
    mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])
    mse = mean_squared_error(evaluation_df['y'], evaluation_df['yhat'])
    rmse = mse ** 0.5  # Root Mean Squared Error

    # Save metrics to file
    metrics_file = 'prophet_forecasting_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")

    print(f"Evaluation metrics saved to '{metrics_file}'.")

def main(csv_file, start_date_str, end_date_str):
    # Convert command-line dates to datetime
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # Load the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'prophet_model.pkl')
    model = load_model(model_file)

    # Load data from CSV file
    data = pd.read_csv(csv_file)
    print(f"Data loaded from {csv_file}. Number of rows: {len(data)}, Number of columns: {len(data.columns)}")

    # Preprocess the data
    future, regressor_columns = preprocess_data(data, start_date, end_date)

    # Forecast and save results
    forecast_output_file = os.path.join(script_dir, 'prophet_forecast.csv')
    forecast = forecast_and_save(model, future, forecast_output_file)

    # Evaluate forecast if target variable exists
    if 'y' in data.columns:
        evaluate_forecast(data, forecast, start_date, end_date)
    else:
        print("PriceHU wasn't found in the provided CSV, so metrics of how well the forecasting did cannot be calculated.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <csv_file> <start_date> <end_date>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])