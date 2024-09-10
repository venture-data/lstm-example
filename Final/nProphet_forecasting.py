import sys
import pandas as pd
import torch
import os
from neuralprophet import NeuralProphet
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Get command-line arguments
input_file = sys.argv[1]  # Path to the input CSV file
forecast_start_date = pd.to_datetime(sys.argv[2])  # Start date for forecasting
forecast_end_date = pd.to_datetime(sys.argv[3])  # End date for forecasting

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'neuralprophet_model.pth')  # Path to the saved .pth file

# Load the entire NeuralProphet model from the .pth file
print("Loading the NeuralProphet model from the .pth file...")
model = torch.load(model_file)  # Use torch.load() to load the entire model
print("Model loaded successfully.")

# Load the input CSV file
print(f"Loading dataset from {input_file}...")
df = pd.read_csv(input_file)
print(f"Dataset loaded with {len(df)} rows.")

# Convert 'Date' column to datetime format and rename columns as needed
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y']  # Ensure 'y' is the target column for forecasting

# Prepare the future DataFrame for predictions
print(f"Preparing future dataframe for predictions from {forecast_start_date} to {forecast_end_date}...")
df_future = df[(df['ds'] >= forecast_start_date) & (df['ds'] <= forecast_end_date)].copy()
print(f"Future dataframe prepared with {len(df_future)} rows.")

# Make predictions
print("Making predictions...")
forecast = model.predict(df_future)
print("Predictions completed.")

# Save the forecast to a CSV file
forecast_output_file = os.path.join(script_dir, 'nProphet_forecast.csv')
forecast.to_csv(forecast_output_file, index=False)
print(f"Forecast results saved to {forecast_output_file}.")

print("Forecasting process completed successfully.")