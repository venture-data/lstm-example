import sys
import pandas as pd
from prophet import Prophet  # Updated import for Prophet
import pickle

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
start_date = sys.argv[2]  # Start date for forecasting
end_date = sys.argv[3]  # End date for forecasting

print(f"Received arguments:\nCSV File: {csv_file}\nStart Date: {start_date}\nEnd Date: {end_date}")

# Load the saved Prophet model
model_file = 'trained_prophet_model.pkl'
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

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
print("Converted 'Date' column to datetime format.")

# Check for NaT (Not a Time) values in the 'Date' column
num_nat = data['Date'].isna().sum()
print(f"Number of NaT values in 'Date' column after conversion: {num_nat}")

# Filter data based on the provided date range
future_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
print(f"Filtered data to the date range from {start_date} to {end_date}. Number of rows after filtering: {len(future_data)}")

# Prepare future dataframe for forecasting
future_data.rename(columns={'Date': 'ds'}, inplace=True)
print(f"Prepared data for forecasting. First few rows:\n{future_data.head()}")

# Check if regressor columns are present in the data
regressor_columns = [col for col in data.columns if col not in ['Date', 'PriceHU']]
missing_regressors = [reg for reg in regressor_columns if reg not in future_data.columns]
if missing_regressors:
    print(f"Warning: Missing regressors in future data: {missing_regressors}")

# Forecast
print("Starting forecast...")
forecast = model.predict(future_data)
print("Forecasting completed.")

# Save forecast results
forecast_file = "forecast_results.csv"
forecast.to_csv(forecast_file, index=False)
print(f"Forecast results saved to '{forecast_file}'.")