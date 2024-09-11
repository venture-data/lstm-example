import sys
import pandas as pd
from prophet import Prophet
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
start_date = pd.to_datetime(sys.argv[2])  # Convert Start date to datetime
end_date = pd.to_datetime(sys.argv[3])  # Convert End date to datetime

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'prophet_model.pkl') 

try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded Prophet model from '{model_file}'.")
except FileNotFoundError:
    print(f"Error: Trained model file '{model_file}' not found.")
    sys.exit(1)

# Load data from CSV file
data = pd.read_csv(csv_file)
print(f"Data loaded from {csv_file}. Number of rows: {len(data)}, Number of columns: {len(data.columns)}")

# Convert 'Date' column to datetime format and rename columns as needed
data['ds'] = pd.to_datetime(data['Date']).dt.round('h')

# Check if 'PriceHU' exists in the columns
is_PriceHU = 'PriceHU' in data.columns

if is_PriceHU:
    print("'PriceHU' column found in data. Setting it as the target variable 'y'.")
    data['y'] = data['PriceHU']  # Ensure 'y' is the target column for forecasting
    data = data.drop(columns=['Date', 'PriceHU'])
else:
    data = data.drop(columns=['Date'])
    

# Ensure 'ds' is in datetime format and rounded to the nearest hour
data['ds'] = pd.to_datetime(data['ds'], errors='coerce').dt.round('h')
print("Converted 'ds' column to datetime format and rounded to the nearest hour.")

# Check for NaT (Not a Time) values in the 'ds' column and remove them
num_nat = data['ds'].isna().sum()
if num_nat > 0:
    print(f"Number of NaT values in 'ds' column after conversion: {num_nat}. Removing these rows...")
    data = data.dropna(subset=['ds'])
    print(f"Removed rows with NaT values. Remaining rows: {len(data)}.")

# Filter data to only include the required prediction dates
future = data[(data['ds'] >= start_date) & (data['ds'] <= end_date)].copy()

# Ensure the future DataFrame is not empty
if future.empty:
    print("Error: The future DataFrame is empty after filtering. Check the date range.")
    sys.exit(1)

# Define regressor columns used during training
regressor_columns = [col for col in data.columns if col not in ['ds', 'y']]  # Define all columns used as regressors

# Ensure the 'is_weekend' and other regressors are correct
# future['is_weekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)  # Recalculate is_weekend if needed

# Fill missing regressor values in the future DataFrame
future[regressor_columns] = future[regressor_columns].ffill().bfill()

print(f"Prepared future DataFrame for forecasting. First few rows:\n{future.head()}")

# Check for NaN values after filling
if future[regressor_columns].isna().sum().sum() > 0:
    print("Warning: There are still NaN values in the regressors after filling.")
    future[regressor_columns] = future[regressor_columns].fillna(0)  # Final fallback to zero

# Forecast
print("Starting forecast...")
forecast = model.predict(future)
print("Forecasting completed.")

# Save the forecast to a CSV file
forecast_output_file = os.path.join(script_dir, 'prophet_forecast.csv')
forecast.to_csv(forecast_output_file, index=False)
print(f"Forecast results saved to {forecast_output_file}.")

if is_PriceHU:
    # Evaluate the forecast
    print("Evaluating the forecast...")
    actual_data = data[(data['ds'] >= start_date) & (data['ds'] <= end_date)][['ds', 'y']].copy()
    forecast_results = forecast[['ds', 'yhat']].copy()
    evaluation_df = pd.merge(actual_data, forecast_results, on='ds', how='inner')

    # Calculate metrics
    mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])
    mse = mean_squared_error(evaluation_df['y'], evaluation_df['yhat'])
    rmse = mse ** 0.5  # Calculate RMSE as the square root of MSE

    metrics_file = 'prophet_forecasting_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")

    print(f"Evaluation metrics saved to '{metrics_file}'.")
else:
    print("PriceHU wasn't found in the provided CSV, so metrics of how well the forecasting did cannot be calculated.")